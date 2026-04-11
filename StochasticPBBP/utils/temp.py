class StationaryMarkov(nn.Module):
    def __init__(self,
                 observation_template: TensorDict,
                 action_template: TensorDict,
                 action_space: Any,
                 hidden_sizes: Tuple[int, ...]=(64, 64),
                 init_weights_fn: str='xavier') -> None:
        super().__init__()
        if not observation_template:
            raise ValueError('observation_template must contain at least one tensor.')
        if not action_template:
            raise ValueError('action_template must contain at least one tensor.')

        self.observation_specs = self._build_observation_specs(observation_template)
        self.action_specs = self._build_action_specs(action_template, action_space)
        self.device = self.observation_specs[0]['device']
        self.dtype = torch.float32

        layers = []
        input_dim = sum(spec['numel'] for spec in self.observation_specs)
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.Tanh())
            input_dim = hidden_size
        output_dim = sum(spec['numel'] for spec in self.action_specs)
        layers.append(nn.Linear(input_dim, output_dim))
        self.network = nn.Sequential(*layers)
        # Default to Xavier init because the network uses Tanh hidden layers.
        self._initialize_network(init_weights_fn)

    @staticmethod
    def _as_tensor(value: Any) -> torch.Tensor:
        return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)

    @classmethod
    def _build_observation_specs(cls, observation_template: TensorDict) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        for (name, template) in observation_template.items():
            tensor = cls._as_tensor(template)
            specs.append({
                'name': name,
                'shape': tuple(tensor.shape),
                'numel': int(tensor.numel()),
                'device': tensor.device,
            })
        return specs

    @classmethod
    def _build_action_specs(
        cls,
        action_template: TensorDict,
        action_space: Any,
    ) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        for (name, template) in action_template.items():
            tensor = cls._as_tensor(template)
            if not tensor.dtype.is_floating_point:
                raise ValueError(
                    f'StationaryMarkov supports only floating-point actions, got {name} '
                    f'with dtype {tensor.dtype}.'
                )
            lower_bound, upper_bound = cls._resolve_action_bounds(name, action_space, tensor)
            specs.append({
                'name': name,
                'shape': tuple(tensor.shape),
                'numel': int(tensor.numel()),
                'dtype': tensor.dtype,
                'device': tensor.device,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
            })
        return specs

    @staticmethod
    def _resolve_action_bounds(
        action_name: str,
        action_space: Any,
        reference: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        spaces = getattr(action_space, 'spaces', None)
        if spaces is None or action_name not in spaces:
            raise ValueError(
                f'Action space does not contain a Box for action {action_name!r}.'
            )
        box = spaces[action_name]
        if not hasattr(box, 'low') or not hasattr(box, 'high'):
            raise ValueError(
                f'Action space entry for {action_name!r} must define low/high bounds.'
            )
        lower_bound = _as_tensor(box.low, dtype=reference.dtype, device=reference.device)
        upper_bound = _as_tensor(box.high, dtype=reference.dtype, device=reference.device)
        if lower_bound.numel() == 1:
            lower_bound = lower_bound.expand(reference.shape)
        if upper_bound.numel() == 1:
            upper_bound = upper_bound.expand(reference.shape)
        if tuple(lower_bound.shape) != tuple(reference.shape):
            raise ValueError(
                f'Lower bounds for action {action_name!r} must match shape {tuple(reference.shape)}, '
                f'got {tuple(lower_bound.shape)}.'
            )
        if tuple(upper_bound.shape) != tuple(reference.shape):
            raise ValueError(
                f'Upper bounds for action {action_name!r} must match shape {tuple(reference.shape)}, '
                f'got {tuple(upper_bound.shape)}.'
            )
        if torch.any(lower_bound > upper_bound):
            raise ValueError(f'Action space bounds for {action_name!r} contain low > high.')
        return lower_bound.clone(), upper_bound.clone()

    def _flatten_observation(self, observation: TensorDict) -> torch.Tensor:
        flat_parts: List[torch.Tensor] = []
        for spec in self.observation_specs:
            name = spec['name']
            if name not in observation:
                raise KeyError(f'Missing observation fluent <{name}>.')
            tensor = self._as_tensor(observation[name]).to(device=spec['device'])
            if tuple(tensor.shape) != spec['shape']:
                raise ValueError(
                    f'Observation <{name}> must have shape {spec["shape"]}, '
                    f'got {tuple(tensor.shape)}.'
                )
            flat_parts.append(tensor.to(dtype=self.dtype).reshape(-1))
        return torch.cat(flat_parts, dim=0)

    def _pack_bounded_actions(self, flat_action: torch.Tensor) -> TensorDict:
        actions: TensorDict = {}
        start = 0
        for spec in self.action_specs:
            end = start + spec['numel']
            raw_action = flat_action[start:end].reshape(spec['shape'])
            bounded_action = self._apply_action_constraints(raw_action, spec)
            actions[spec['name']] = bounded_action.to(dtype=spec['dtype'], device=spec['device'])
            start = end
        return actions

    def _apply_action_constraints(self,
                                  raw_action: torch.Tensor,
                                  spec: Dict[str, Any]) -> torch.Tensor:
        lower_bound = spec['lower_bound'].to(dtype=raw_action.dtype, device=raw_action.device)
        upper_bound = spec['upper_bound'].to(dtype=raw_action.dtype, device=raw_action.device)
        bounded_action = raw_action.clone()

        has_lower = torch.isfinite(lower_bound)
        has_upper = torch.isfinite(upper_bound)
        both_bounded = has_lower & has_upper
        lower_only = has_lower & ~has_upper
        upper_only = ~has_lower & has_upper

        if torch.any(both_bounded):
            normalized_action = torch.sigmoid(raw_action)
            scaled_action = lower_bound + (upper_bound - lower_bound) * normalized_action
            bounded_action = torch.where(both_bounded, scaled_action, bounded_action)
        if torch.any(lower_only):
            lower_bounded_action = lower_bound + F.softplus(raw_action)
            bounded_action = torch.where(lower_only, lower_bounded_action, bounded_action)
        if torch.any(upper_only):
            upper_bounded_action = upper_bound - F.softplus(raw_action)
            bounded_action = torch.where(upper_only, upper_bounded_action, bounded_action)

        return bounded_action

    def _initialize_network(self, init_weights_fn: str) -> None:
        normalized_name = init_weights_fn.strip().lower()
        if normalized_name in {'kaiming', 'jax'}:
            initializer = self._init_weights_kaiming
        elif normalized_name == 'xavier':
            initializer = self._init_weights_xavier
        else:
            raise ValueError(
                f'Unsupported init_weights_fn={init_weights_fn!r}. '
                "Expected 'kaiming', 'jax', or 'xavier'."
            )
        self.network.apply(initializer)

    @staticmethod
    def _init_weights_xavier(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _init_weights_kaiming(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self,
                observation: TensorDict,
                step: Optional[int]=None,
                policy_state: Any=None) -> TensorDict:
        del step, policy_state
        flat_observation = self._flatten_observation(observation)
        flat_action = self.network(flat_observation)
        return self._pack_bounded_actions(flat_action)

    def sample_action(self, observation: TensorDict) -> TensorDict:
        with torch.no_grad():
            return self.forward(observation, step=None, policy_state=None)
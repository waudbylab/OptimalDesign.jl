# OptimalDesign.jl — Development Notes

## Logging convention

Use Julia's standard logging macros (`@info`, `@debug`, `@warn`) throughout.
No `verbose` keyword arguments — callers control visibility via the logging system.

- **`@info`** — high-level process milestones: experiment start/complete, exchange
  algorithm phase transitions, convergence status, final results.
- **`@debug`** — per-iteration detail: FW gap, support size, step transfers,
  per-observation updates in adaptive loops.
- **`@warn`** — singular FIM, resampling triggers, non-convergence.

To see debug output in example scripts or the REPL:

```julia
ENV["JULIA_DEBUG"] = OptimalDesign
```

## Code organisation

- **All type definitions go in `types.jl`** — structs, abstract types, type aliases.
  Other files contain methods, constructors, and logic only.

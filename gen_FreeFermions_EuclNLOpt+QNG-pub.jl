begin
  ENV["MKL_DEBUG_CPU_TYPE"] = "5"
  ENV["BLIS_NUM_THREADS"] = "4"
  using LinearAlgebra, Random, Printf, Statistics
  using NLopt
  using Zygote, DifferentiationInterface, ADTypes, ForwardDiff, ChainRules, ChainRulesCore
  # using OptimizationOptimJL, OptimizationIpopt
  using Optim, NLSolversBase
  using F_utilities
  using HDF5
  using BLISBLAS
end

LinearAlgebra.BLAS.set_num_threads(Threads.nthreads())
BLISBLAS.set_num_threads(Threads.nthreads())

include("free_fermions_simulation.jl")

# ════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these constants to change the run
# ════════════════════════════════════════════════════════════════════════

# Select ansatz:
#   :Rxx_Rz    — 2-param NN,  angles bounded to [-π, π]  (e^{-iJ·Rxx} · e^{-ih·Rz})
#   :TFI_only  — 2-param NN,  unbounded                  (e^{-i(J·XX + h·Z)})
#   :TFI_Rz    — 3-param NN,  third angle bounded [-π,π] (e^{-i(tJ·XX+h·Z)} · e^{-iθ·Rz})
#   :brickwall — 4-param NN,  unbounded staggered layers  (even + odd bonds) ### Depreciated, test purposes only

# const ANSATZ = :brickwall
const ANSATZ = Symbol(ARGS[1])

# System sizes and circuit depths to sweep over
# const Ns = [8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 128, 256, 512]
# const Ns = [8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 128, 256, 512]
const Ns = [8, 12,]

# const Ds = [1:12..., 14:2:24...]
const Ds = 1:2
# const Ds = 4:12
# const Ds = [14:2:24...]


# Penalty sweep: [true, false] runs penalty first to warm-start, then the
# clean run; [false] skips the penalty stage entirely; [true] only penalty.
# const PENALTY_SWEEP = [false]

# When overwrite = false and the output file already exists:
#   - existing parameters are loaded into network_params for warm-starting
#   - the optimisation loop is skipped (continue)
# When overwrite = true the optimisation always runs, even if a file exists.
const overwrite = if length(ARGS) >=2
  parse(Bool, ARGS[2])
else
  true
end

# ════════════════════════════════════════════════════════════════════════
# PER-ANSATZ CONFIGURATION TABLE  (do not normally edit below)
# ════════════════════════════════════════════════════════════════════════

const _CONFIGS = (
    Rxx_Rz = (
        fun         = vg_NN_Rxx_Rz,
        fun_vgm     = vgm_NN_Rxx_Rz,
        label       = "Rxx_Rz",
        n_params    = 2,                         # parameters per layer
        lb_tile     = [-1π,   -1π  ],              # repeated D times for lower bound
        ub_tile     = [+1π,   +1π  ],              # repeated D times for upper bound
        wrap        = true,                      # apply rem2pi after loading/seeding
        init_single = () -> rem2pi.(0.05 .* rand(2), RoundNearest),
    ),
    
    TFI_only = (
        fun         = vg_NN_TFI_only,
        fun_vgm     = vgm_NN_TFI_only,
        label       = "TFI_only",
        n_params    = 2,
        lb_tile     = [-Inf, -Inf],
        ub_tile     = [+Inf, +Inf],
        wrap        = false,
        init_single = () -> 0.05 .* randn(2),
    ),
  
    kitaev_only = (
        fun         = vg_NN_Kitaev_only,
        fun_vgm     = vgm_NN_Kitaev_only,
        label       = "kitaev_only",
        n_params    = 3,
        lb_tile     = [-Inf, -Inf, -Inf],
        ub_tile     = [+Inf, +Inf, +Inf],
        wrap        = false,
        init_single = () -> 0.05 .* randn(3),
    ),
    kitaev_HVA = (
        fun         = vg_NN_Kitaev_HVA,
        fun_vgm     = vgm_NN_Kitaev_HVA,
        label       = "kitaev_HVA",
        n_params    = 3,
        lb_tile     = [-Inf, -Inf, -π],
        ub_tile     = [+Inf, +Inf, +π],
        wrap        = false,
        init_single = () -> 0.05 .* randn(3),
    ),
    TFI_EXP_only = (
        fun         = vg_LR_EXP_TFI_only,
        fun_vgm     = vgm_LR_EXP_TFI_only,
        label       = "TFI_EXP_only",
        n_params    = 3,
        lb_tile     = [-1.0, -Inf, -Inf],
        ub_tile     = [+1.0, +Inf, +Inf],
        wrap        = false,
        init_single = () -> vcat(0.05 .* randn(3)),
    ),
    TFI_POW_only = (
        fun         = vg_LR_POW_TFI_only,
        fun_vgm     = vgm_LR_POW_TFI_only,
        label       = "TFI_POW_only",
        n_params    = 3,
        lb_tile     = [+0.0, -Inf, -Inf],
        ub_tile     = [+Inf, +Inf, +Inf],
        wrap        = false,
        init_single = () -> vcat(2.0 .+ 0.1 .* randn(1), 0.05 .* randn(2)),
    ),

    TFI_EXP_HVA = (
        fun         = vg_EXP_HVA,
        fun_vgm     = vgm_EXP_HVA,
        label       = "TFI_EXP_HVA",
        n_params    = 3,
        lb_tile     = [-1.0, -Inf, -Inf],
        ub_tile     = [+1.0, +Inf, +Inf],
        wrap        = false,
        init_single = () -> vcat(0.05 .* randn(3)),
),
    TFI_POW_HVA = (
        fun         = vg_POW_HVA,
        fun_vgm     = vgm_POW_HVA,
        label       = "TFI_POW_HVA",
        n_params    = 3,
        lb_tile     = [+0.0, -Inf, -Inf],
        ub_tile     = [+Inf, +Inf, +Inf],
        wrap        = false,
        init_single = () -> vcat(2.0 .+ 0.1 .* randn(1), 0.05 .* randn(2)),
    )
)

const cfg = getfield(_CONFIGS, ANSATZ)

# ════════════════════════════════════════════════════════════════════════
# PRECONDITIONER VARIANTS — previously-computed results used for warm-starts
# Keys must match _CONFIGS keys above.
# ════════════════════════════════════════════════════════════════════════


E_OBC(N) = 1.0 - csc(π / (4N + 2))

# network_params[(N, D)] holds the best known parameters for system size N
# and circuit depth D.  Pre-populate with known good starting points here
# (or leave empty to start from random initialisations).
network_params = Dict{Tuple{Int,Int}, Vector{Float64}}(

)


nP = cfg.n_params

# for with_penalty in PENALTY_SWEEP
const with_penalty = false

for D in Ds
  println("\n══════════════════════ D = $D  |  penalty = $with_penalty ══════════════════════")

  for (iN, N) in enumerate(Ns)
    if N > 32 && D > 12
      continue
    end
    

    # ── Output paths ───────────────────────────────────────────────────
    # solver = Optim.LBFGS(m=1;)
    # solver = Optim.ConjugateGradient(eta = 0.2, linesearch = Optim.LineSearches.MoreThuente())
    # solver = Optim.BFGS(; alphaguess = Optim.LineSearches.InitialStatic(),
    #    linesearch = Optim.LineSearches.HagerZhang(),
    #    initial_invH = nothing,
    #    initial_stepnorm = nothing,
    #    manifold = Flat()
    # )

    # solver = Optim.ConjugateGradient()

    solver = :LD_LBFGS

    solver_string =  split(split(string(solver),'(')[1],'{')[1]
    storepath = joinpath("./new_data", "Optim_QNG_and_Euclidian_$(solver_string)_$(cfg.label)", "D_$(D)")

    mkpath(storepath)
    filename  = joinpath(storepath, "Store_QNG_and_Euclidian_$(solver_string)_NN_$(cfg.label)_N$(N)_D$(D)_withPenalty_$(with_penalty).hdf5")
    
    println("── N = $N  →  $filename")

    # ── Target Hamiltonian ─────────────────────────────────────────────
    GS0                 = psi_up(N)
    H_target            = TFI_Hamiltonian(N, 1.0, 1.0)
    HD_target, U_target = Fu.Diag_h(H_target)
    Γ_target            = Fu.GS_gamma(HD_target, U_target)
    E_GS_target         = Fu.Energy(Γ_target, (HD_target, U_target))
    @printf("   E_GS = %.10f   E_OBC = %.10f\n", E_GS_target, E_OBC(N))

    # ── Load existing files and warm-start with the best parameters ────
    # Collect (energy, params) from each file that exists, then keep the
    # set with the lowest energy.
    let candidates = Tuple{Float64, Vector{Float64}}[]
      for (tag, fpath) in [("current", filename)]
        fpath === nothing && continue
        isfile(fpath) || continue
        try
          h5open(fpath, "r") do fh
            e = read(fh, "minimal Energy")
            p = read(fh, "minimal network parameters")
            push!(candidates, (e, p))
            println("   Found $tag file  E = $e  ($fpath)")
          end
        catch
        end
      end
      if !isempty(candidates)
        best_e, best_p = candidates[argmin(first.(candidates))]
        network_params[(N, D)] = best_p
        @printf("   → warm-starting from best candidate  E = %.10f\n", best_e)
      end
      # continue ## jump over current file if already exists
    end

    # isfile(filename) && continue

    # ── Initial guess ──────────────────────────────────────────────────
    # Priority:
    #  1. exact (N, D) key
    #  2. same D, largest available N' ≤ N  (params have the same length)
    #  3. largest available D' ≤ D (then largest N' ≤ N): pad the missing
    #     layers by interjecting init_single() symmetrically in the middle
    #  4. fully random initialisation
    x0 = if haskey(network_params, (N, D))
      v = copy(network_params[(N, D)])
      cfg.wrap ? rem2pi.(v, RoundNearest) : v
    else
      # ── fallback 2: same D, largest N' ≤ N ──────────────────────────
      same_D_idx = findlast(i -> Ns[i] <= N && haskey(network_params, (Ns[i], D)), eachindex(Ns))
      if same_D_idx !== nothing
        v = copy(network_params[(Ns[same_D_idx], D)])
        println("   Warm-start: (N'=$(Ns[same_D_idx]), D=$D) → (N=$N, D=$D)")
        cfg.wrap ? rem2pi.(v, RoundNearest) : v
      else
        # ── fallback 3: largest D' ≤ D, then largest N' ≤ N ─────────
        best = nothing
        for D′ in reverse(filter(d -> d <= D, collect(Ds)))
          idx = findlast(i -> Ns[i] <= N && haskey(network_params, (Ns[i], D′)), eachindex(Ns))
          if idx !== nothing
            best = (Ns[idx], D′)
            break
          end
        end
        if best !== nothing
          N′, D′ = best
          v      = copy(network_params[(N′, D′)])
          n_extra = D - D′
          half    = ANSATZ == :TFI_EXP_HVA ? length(v) : (D′ ÷ 2) * nP          # split existing layers symmetrically
          filler  = reduce(vcat, [cfg.init_single() for _ in 1:n_extra])
          raw     = vcat(v[1:half], filler, v[half+1:end])
          println("   Warm-start: (N'=$N′, D'=$D′) → (N=$N, D=$D), interjected $n_extra layer(s)")
          cfg.wrap ? rem2pi.(raw, RoundNearest) : raw
        else
          # ── fallback 4: fully random ───────────────────────────────
          reduce(vcat, [cfg.init_single() for _ in 1:D])
        end
      end
    end

    @show x0

    # x0[1:2:end] .*= -1

    # x0 = reduce(vcat, [cfg.init_single() for _ in 1:D])
    # ── Parameter bounds ───────────────────────────────────────────────
    lower_bound = repeat(cfg.lb_tile, D)
    upper_bound = repeat(cfg.ub_tile, D)

    params = (GS0, N, D, (HD_target, U_target), false, false)

    Eref = E_OBC(N)
    


    adaptive_dopri5_ng_box!(
      x0,
      -0.01,
      cfg.fun_vgm,
      params,
      lower_bound,
      upper_bound;
      adapt_tol = true,
      max_iter = 500,
      tol = 5e-5,
      ref_val = Eref,
      # descent_tol = 1
      descent_tol = 1e-9,
      descent_patience = 5,
      climb_patience = 5,
    )


    # θs = copy(x0)

    function my_objective_fn(x::Vector, grad::Vector; params=params, refVal=Eref)
      f, ∇f = cfg.fun(x, params) ## no QNG variant as BFGS assumes flat geometry in variable space
      if length(grad) > 0
        grad .= ∇f
      end
      diff = f - refVal
      @printf(" f=%+1.16f | ΔE=%1.16e | Δe₀=%1.16e | ‖∇f‖=%1.12e\n",
                # state.f_x, state.f_x - E_ref, state.f_x/N + 4/π, norm(state.g_x))
                f, diff, diff/N, norm(grad))
      flush(stdout)
      # push!(trace, copy(x) => f)
      return f
    end

    @show my_objective_fn(x0, similar(x0))

    opt2 = NLopt.Opt(:LD_LBFGS, cfg.n_params * D)
    opt2.vector_storage = cfg.n_params * D ## estimate the full hessian
    NLopt.min_objective!(opt2, my_objective_fn)
    
    NLopt.lower_bounds!(opt2, lower_bound)
    NLopt.upper_bounds!(opt2, upper_bound)
    NLopt.xtol_rel!(opt2, 1e-12)
    NLopt.ftol_rel!(opt2, 1e-12)
    NLopt.maxtime!(opt2, 60*60*12)
    NLopt.maxeval!(opt2, 999)
    NLopt.stopval!(opt2, E_OBC(N))

    E_min, x_min, return_value = NLopt.optimize(opt2, x0)
    final_params = copy(x_min)
    
    @show final_params
    # ── Warm-start seeding for (N, D+1) and (Ns[iN+1], D) ─────────────
    # Insert `nP` random parameters at the midpoint of the parameter vector
    # to initialise the D+1 circuit (D==1 special case: append at the end).
    if !haskey(network_params, (N, D+1))
      interject = nP * (D ÷ 2)
      seed      = cfg.init_single()
      np1       = vcat(final_params[1:interject], seed, final_params[interject+1:end])
      network_params[(N, D+1)] = cfg.wrap ? rem2pi.(np1, RoundNearest) : np1
    end
    if iN < length(Ns)
      network_params[(Ns[iN+1], D)] = final_params
    end

    # ── Evaluate final state and save ─────────────────────────────────
    # params_out = (GS0, N, D, (HD_target, U_target), true, false)
    final_energy, gradient, state_final = cfg.fun(final_params, params, return_state=true)

    observables = single_eval_Ising(
      state_final,
      U_target,
      Γ_target,
      final_energy,
      final_params,
      N,
      D,
    )

    # Persist every publication observable as a root-level dataset in this
    # run's HDF5 file. The NamedTuple field names become the dataset names.
    h5open(filename, "w") do file
      for name in propertynames(observables)
        write(file, string(name), getproperty(observables, name))
      end
      flush(file)
    end

    @printf(" →→→  DONE: D = %d, N = %d\tE = %.10f  (E_OBC = %.10f)\n", D, N, final_energy, E_OBC(N))
  end # N

  println("══ D = $D done ══\n")
end # D
# end # with_penalty

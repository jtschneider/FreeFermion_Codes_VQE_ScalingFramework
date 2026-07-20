# Focused free-fermion simulation library.


using LinearAlgebra
using Random
using Printf
using Statistics
using BandedMatrices
using Zygote
using ForwardDiff
using ChainRulesCore
using LsqFit
using HDF5
import F_utilities as Fu

"""
    _safe_diag_h(H) -> (D, U)

Wrapper around `Fu.Diag_h` that retries with progressively larger Hermitian
perturbations when LAPACK's Schur decomposition fails to converge.
Needed for Hamiltonians with alternating-zero couplings (brickwall ansatz) whose
skew form causes QR non-convergence even under mode-1 perturbation.
"""
function _safe_diag_h(H::AbstractMatrix)
    n = size(H, 1)
    for (attempt, δ) in enumerate((0.0, 1e-12, 1e-10, 1e-8, 1e-6))
        try
            if δ == 0.0
                return Fu.Diag_h(H, 1)
            else
                R = δ .* (randn(n, n) .+ im .* randn(n, n))
                return Fu.Diag_h(H .+ (R .+ R') ./ 2, 1)
            end
        catch e
            e isa LinearAlgebra.LAPACKException || rethrow(e)
            attempt == 5 && rethrow(e)
            @debug "_safe_diag_h: LAPACK failed (attempt $attempt), retrying with δ=$(δ*100)"
        end
    end
end


function Evolve_exp_AD(M, H, t)
    U = exp(im * 2 * t .* H)
    # M_new = U * M * U'
    return Hermitian(U * M * U')
end


# ------------------------------------------------------------------------------
# Second-order-AD-compatible matrix exponential via augmented matrix trick.
# Registers a LinearAlgebra.exp method for Matrix{Complex{<:ForwardDiff.Dual}}
# so that ForwardDiff.hessian / ForwardDiff.jacobian ∘ Zygote.gradient work
# through Evolve_exp_AD without hitting exp!'s type restriction.
#
# Mathematical basis: for Y = exp(A) and tangent direction E,
#   D_E exp(A) = exp(Ã)[1:n, n+1:2n]   where  Ã = [A  E; 0  A]
# For N partial directions, we call exp on a 2n×2n plain-float matrix N times.
# For nested duals (ForwardDiff.hessian), the method recurses automatically.
# ------------------------------------------------------------------------------
function LinearAlgebra.exp(A::AbstractMatrix{Complex{D}}) where {D <: ForwardDiff.Dual}
    n   = size(A, 1)
    T   = ForwardDiff.tagtype(D)
    V   = ForwardDiff.valtype(D)
    nP  = ForwardDiff.npartials(D)

    # Primal matrix — may still contain inner Dual layers (nested / 2nd-order)
    A₀ = complex.(ForwardDiff.value.(real.(A)), ForwardDiff.value.(imag.(A)))
    Y  = exp(A₀)   # dispatches recursively if V <: Dual, else plain LinearAlgebra.exp

    # One augmented exp per partial direction k
    dY = ntuple(nP) do k
        Ȧₖ = complex.(
            map(a -> ForwardDiff.partials(a, k), real.(A)),
            map(a -> ForwardDiff.partials(a, k), imag.(A)),
        )
        Ã    = [A₀  Ȧₖ; zero(A₀)  A₀]
        expÃ = exp(Ã)   # 2n×2n; also recurses if V <: Dual
        expÃ[1:n, n+1:2n]
    end

    # Reassemble Matrix{Complex{D}}
    result = similar(A)
    @inbounds for j in 1:n, i in 1:n
        rp = ntuple(k -> real(dY[k][i,j]), nP)
        ip = ntuple(k -> imag(dY[k][i,j]), nP)
        result[i,j] = Complex(
            ForwardDiff.Dual{T,V,nP}(real(Y[i,j]), ForwardDiff.Partials{nP,V}(rp)),
            ForwardDiff.Dual{T,V,nP}(imag(Y[i,j]), ForwardDiff.Partials{nP,V}(ip)),
        )
    end
    return result
end


# ------------------------------------------------------------------------------
# ChainRules rrule for Evolve_exp_AD — enables second-order AD via
# ForwardDiff.jacobian ∘ Zygote.gradient (forward-over-reverse Hessian).
#
# The pullback for A = 2i·t·H uses the Fréchet adjoint of exp:
#   ΔA = exp([A†, ΔU; 0, A†])[1:n, n+1:2n]
# which dispatches to the custom ForwardDiff exp overload above when the
# primal values are Duals, making the pullback itself ForwardDiff-differentiable.
# ------------------------------------------------------------------------------
function ChainRulesCore.rrule(::typeof(Evolve_exp_AD), M, H, t)
    n  = size(H, 1)
    A  = 2im * t .* H
    U  = exp(A)
    Y  = Hermitian(U * M * U')

    function Evolve_exp_AD_pullback(ΔY_raw)
        ΔY = ΔY_raw isa ChainRulesCore.AbstractZero ? zero(Matrix(Y)) : Matrix(ΔY_raw)

        # Pullback for M: from Y = U·M·U†
        ΔM = U' * ΔY * U

        # Sensitivity for U: from Y = (U·M)·U† (product rule with adjoint)
        ΔU = ΔY * U * M' + ΔY' * U * M

        # Pullback for A via Fréchet adjoint of exp at A:
        #   (D_A exp)*(ΔU) = exp([A†, ΔU; 0, A†])[1:n, n+1:2n]
        Ã  = [A'  ΔU; zero(ΔU)  A']
        ΔA = exp(Ã)[1:n, n+1:2n]

        # Chain rule for A = 2im·t·H
        ΔH = conj(2im * t) .* ΔA
        Δt = real(tr(ΔA' * (2im .* H)))

        return ChainRulesCore.NoTangent(), ΔM, ΔH, Δt
    end

    return Y, Evolve_exp_AD_pullback
end


function fit_boundary_cutoff_robust(S_profile, N; r_tol = 0.015)
    if all(isnan, S_profile)
        return (; c_eff=NaN, s₁=NaN, a_osc=NaN, ξ_ent=NaN,
                  plateau_start=0, plateau_length=0,
                  S_plateau=NaN, residual=NaN, param=[NaN, NaN, NaN])
    end

    N_A = length(S_profile)
    ℓs = Float64.(collect(1:N_A))
    chord = (2N/π) .* sin.(π .* ℓs ./ N)

    # ── Step 1: Separate even/odd and average to get smooth profile ───
    S_even = S_profile[2:2:end]
    S_odd  = S_profile[1:2:end]

    # Interpolate both onto all sites to get a smooth envelope
    S_smooth = copy(Float64.(S_profile))
    for i in 1:N_A
        # Average with neighbors to suppress staggering
        lo = max(1, i-1)
        hi = min(N_A, i+1)
        S_smooth[i] = mean(S_profile[lo:hi])
    end

    # ── Step 2: Find plateau from smoothed profile ───────────────────
    S_bulk = mean(S_smooth[max(1, N_A-3):N_A])
    tolerance = r_tol * S_bulk

    plateau_end = N_A
    plateau_start = N_A
    for i in N_A:-1:1
        if abs(S_smooth[i] - S_bulk) < tolerance
            plateau_start = i
        else
            break
        end
    end

    plateau_length = plateau_end - plateau_start + 1

    if plateau_length > 2
        fit_end = max(4, plateau_start - 1)
        S_plateau = mean(S_smooth[plateau_start:end])
        ξ_ent = chord[plateau_start]
    else
        fit_end = N_A
        S_plateau = S_smooth[end]
        ξ_ent = chord[end]
    end

    # ── Step 3: Fit CC + oscillating correction on original data ─────
    # S(ℓ) = (c/6) ln(chord) + s₁ + a_osc * (-1)^ℓ / chord^p
    # Fix p = 1 (leading Ising correction)
    fit_start = 1
    ℓ_fit = ℓs[fit_start:fit_end]
    S_fit = Float64.(S_profile[fit_start:fit_end])

    model_osc(ℓ, p) = (p[1] .* log.((2N/π) .* sin.(π .* ℓ ./ N))
                        .+ p[2]
                        .+ p[3] .* (-1.0).^Int.(ℓ) ./ ((2N/π) .* sin.(π .* ℓ ./ N)))

    # Try oscillating fit first
    fit_ok = true
    local fit
    try
        fit = curve_fit(model_osc, ℓ_fit, S_fit, [0.5/6, 0.0, 0.0])
    catch
        fit_ok = false
    end

    if fit_ok && abs(fit.param[1]) < 1.0  # sanity check on c/6
        c_eff = 6 * fit.param[1]
        s₁ = fit.param[2]
        a_osc = fit.param[3]
        residual = norm(fit.resid) / length(ℓ_fit)
        param = fit.param
    else
        # Fallback: fit on smoothed profile without oscillation
        model_bare(ℓ, p) = p[1] .* log.((2N/π) .* sin.(π .* ℓ ./ N)) .+ p[2]
        S_smooth_fit = S_smooth[fit_start:fit_end]
        fit2 = curve_fit(model_bare, ℓ_fit, S_smooth_fit, [0.5/6, 0.0])
        c_eff = 6 * fit2.param[1]
        s₁ = fit2.param[2]
        a_osc = 0.0
        residual = norm(fit2.resid) / length(ℓ_fit)
        param = [fit2.param..., 0.0]
    end

    return (; c_eff, s₁, a_osc, ξ_ent, plateau_start, plateau_length,
              S_plateau, residual, param)
end


function evaluate_and_save(
    state_final::AbstractMatrix{<:Number},
    H_target::AbstractMatrix{<:Number},
    U_target::AbstractMatrix{<:Number},
    HD_target::AbstractMatrix{<:Real},
    Γ_target::AbstractMatrix{<:Number},
    final_energy::Float64,
    final_params::Vector{Float64},
    N::Int,
    D::Int,
    filename::String
)
    N_A = N ÷ 2

    # ── Legacy observables ───────────────────────────────────────────
    n_k = real(diag(U_target' * state_final * U_target))
    variance_H = hamiltonian_variance(state_final, HD_target, U_target)
    final_entropy = Fu.VN_entropy(Fu.Reduce_gamma(state_final, N_A, 1))
    ES_vals_even, ES_vals_odd = ES_spec(state_final)
    ES_ratios_even, ES_ratios_odd = ES_spec_ratios(ES_vals_even, ES_vals_odd)

    # ── Entanglement profiles S(ℓ) for ℓ = 1, ..., N/2 ──────────────
    S_profile_circuit = [Fu.VN_entropy(Fu.Reduce_gamma(state_final, ℓ, 1)) for ℓ in 1:N_A]
    S_profile_target  = [Fu.VN_entropy(Fu.Reduce_gamma(Γ_target, ℓ, 1)) for ℓ in 1:N_A]

    # ── Entanglement penetration depth from profile fit ──────────────
    efit = fit_boundary_cutoff_robust(S_profile_circuit, N)
    ξ_ent = efit.ξ_ent
    c_eff = efit.c_eff
    plateau_start = efit.plateau_start

    # ── Two-point correlations w.r.t. the centre site ────────────────
    # BdG/Nambu convention used throughout (matches Surace 2022):
    #   Γ[j, l]       = ⟨c†_j c_l⟩         upper-left block
    #   Γ[j, l+N]     = ⟨c†_j c†_l⟩        upper-right (anomalous)
    #   Γ[j+N, l]     = ⟨c_j c_l⟩          lower-left  (anomalous)
    #   Γ[j+N, l+N]   = ⟨c_j c†_l⟩         lower-right
    # We pin one operator at j = N÷2 and let l = j + n run over 1..N,
    # giving n ∈ {-N/2+1, ..., N/2}.  All four 2-point combinations are
    # extracted for both the circuit state and the target state, so the
    # decay can be inspected separately in normal and anomalous channels.
    # idx_c     = N ÷ 2
    # n_offsets = collect((-N÷2 + 1):(N÷2))   # n; site index l = idx_c + n

    # Circuit state
    # G_cdag_c_circuit    = Vector{ComplexF64}(state_final[idx_c,     1:N    ])  # ⟨c†_c c_l⟩
    # G_cdag_cdag_circuit = Vector{ComplexF64}(state_final[idx_c,     N+1:2N ])  # ⟨c†_c c†_l⟩
    # G_c_c_circuit       = Vector{ComplexF64}(state_final[idx_c + N, 1:N    ])  # ⟨c_c c_l⟩
    # G_c_cdag_circuit    = Vector{ComplexF64}(state_final[idx_c + N, N+1:2N ])  # ⟨c_c c†_l⟩

    # # Target state (Gaussian ground state of H_target)
    # G_cdag_c_target     = Vector{ComplexF64}(Γ_target[idx_c,     1:N    ])
    # G_cdag_cdag_target  = Vector{ComplexF64}(Γ_target[idx_c,     N+1:2N ])
    # G_c_c_target        = Vector{ComplexF64}(Γ_target[idx_c + N, 1:N    ])
    # G_c_cdag_target     = Vector{ComplexF64}(Γ_target[idx_c + N, N+1:2N ])

    # # ── Parent Hamiltonian → RG couplings (scalars only) ─────────────
    # Γ_D, U_parent = Fu.Diag_gamma(state_final)
    # ν = map(x -> clamp(x, 0.0, Inf), real.(diag(Γ_D)[N+1:2N]))
    # ν_safe = clamp.(ν, 1e-12, 1 - 1e-12)
    # ε_parent = @. 0.5 * log((1 - ν_safe) / ν_safe)
    # H_D_parent = diagm(vcat(ε_parent, -ε_parent))
    # H_parent = U_parent * H_D_parent * U_parent'

    # A_raw = -H_parent[1:N, 1:N]
    # B_raw = -H_parent[N+1:2N, 1:N]

    # # Normalise: fix kinetic (NN hopping) to match target v_target = 0.5
    # margin = max(2, N ÷ 6)
    # bulk = margin+1 : N-margin
    # v_parent = mean(abs(real(A_raw[i, i+1])) for i in bulk if i+1 ≤ N)

    # H_p_normalisation = 0.5 / v_parent # critical TFI always has J/2 on the off-diagonal

    # H_scaled = H_p_normalisation * H_parent
    # A_p = H_p_normalisation * A_raw
    # B_p = H_p_normalisation * B_raw
    # A_t = -H_target[1:N, 1:N]
    # B_t = -H_target[N+1:2N, 1:N]
    # # δA = A_p - A_t
    # δB = B_p - B_t

    # Bulk-averaged hopping and pairing deviation profiles
    # margin = max(2, N ÷ 6)
    # bulk = margin+1 : N-margin
    # max_r = N ÷ 4

    # δt = [mean(real(δA[i, i+r]) for i in bulk if 1 ≤ i+r ≤ N) for r in 0:max_r]
    # δb = [mean(real(δB[i, i+r]) for i in bulk if 1 ≤ i+r ≤ N) for r in 0:max_r]

    # # RG couplings (v_target = 0.5 for critical TFI)
    # g_mass      = sum(δt)
    # g_curvature = (1/24) * sum(r^4 * δt[r+1] for r in 0:max_r)
    # g_gap       = sum(r * δb[r+1] for r in 0:max_r)
    # g_gap3      = -(1/6) * sum(r^3 * δb[r+1] for r in 0:max_r)

    # Norm of imaginary part of A (Majorana xp block proxy)
    # imA_density = norm(imag.(A_p)) / N

    # Hamiltonian direction overlap (scale-invariant)
    # h_p = H_scaled / norm(H_scaled)
    # h_t = H_target / norm(H_target)
    # overlap = real(tr(h_p' * h_t))

    # ── Print summary ────────────────────────────────────────────────
    @show final_energy, E_OBC(N)
    # @show final_entropy, variance_H
    # @show H_p_normalisation
    # @show g_mass, g_gap, g_curvature, g_gap3
    @show ξ_ent, c_eff, plateau_start
    # @show imA_density
    uptoE = min(length(ES_ratios_even), 22)
    uptoO = min(length(ES_ratios_odd), 22)
    @show ES_ratios_even[1:uptoE]
    @show ES_ratios_odd[1:uptoO]

    @show ES_vals_even[1:uptoE]
    @show ES_vals_odd[1:uptoO]

    # ── Save to HDF5 (no large matrices) ─────────────────────────────
    h5open(filename, "w") do file
        # System parameters
        write(file, "N", N)
        write(file, "number of layers", D)
        write(file, "minimal network parameters", final_params)

        # Energy sector
        write(file, "minimal Energy", final_energy)
        write(file, "variance_H", variance_H)

        # Mode occupations (length 2N vector)
        write(file, "occupation", n_k)

        # Entanglement sector
        write(file, "entanglement entropy optimized state", final_entropy)
        write(file, "entanglement spectrum optimized state N÷2 EVEN", ES_ratios_even)
        write(file, "entanglement spectrum optimized state N÷2 ODD", ES_ratios_odd)

        write(file, "entanglement spectrum values optimized state N÷2 EVEN", ES_vals_even)
        write(file, "entanglement spectrum values optimized state N÷2 ODD", ES_vals_odd)

        write(file, "entanglement profile circuit", S_profile_circuit)
        write(file, "entanglement profile target", S_profile_target)

        # Entanglement profile fit
        write(file, "xi_ent", ξ_ent)
        write(file, "c_eff", c_eff)
        write(file, "plateau_start", plateau_start)

        # Two-point correlations vs offset n from centre site idx_c = N÷2
        # write(file, "twopt n_offsets",            n_offsets)
        # write(file, "twopt G_cdag_c circuit",     G_cdag_c_circuit)
        # write(file, "twopt G_cdag_cdag circuit",  G_cdag_cdag_circuit)
        # write(file, "twopt G_c_c circuit",        G_c_c_circuit)
        # write(file, "twopt G_c_cdag circuit",     G_c_cdag_circuit)
        # write(file, "twopt G_cdag_c target",      G_cdag_c_target)
        # write(file, "twopt G_cdag_cdag target",   G_cdag_cdag_target)
        # write(file, "twopt G_c_c target",         G_c_c_target)
        # write(file, "twopt G_c_cdag target",      G_c_cdag_target)
        write(file, "Gamma_target", state_final)

        # Parent Hamiltonian: scalars only
        # write(file, "parent H normalisation", H_p_normalisation)
        # write(file, "parent H overlap", overlap)
        # write(file, "parent H_p imA density", imA_density)

        # RG couplings (scalars)
        # write(file, "rg g_mass", g_mass)
        # write(file, "rg g_gap", g_gap)
        # write(file, "rg g_curvature", g_curvature)
        # write(file, "rg g_gap3", g_gap3)

        # Coupling deviation profiles (length max_r+1 vectors)
        # write(file, "rg delta_t profile", δt)
        # write(file, "rg delta_b profile", δb)
    end
end


"""
    adaptive_dopri5_box!(θs, h₀, cost_fn, lb, ub; kwargs...)

Adaptive Dormand–Prince 5(4) integrator for projected gradient descent
on a box-constrained domain lb[i] ≤ θs[i] ≤ ub[i], via imaginary time
evolution. h₀ must be negative (descent).

Use `lb[i] = -Inf` and/or `ub[i] = +Inf` for unconstrained coordinates.

Projection is applied to every stage trial point (feasibility) and the
gradient is cone-projected at each stage (no component pointing into an
active wall), consistent with h < 0 step direction.

FSAL: k₇ (cone-projected at the accepted point) is recycled as k₁ of
the next step — only 6 new gradient evaluations per accepted step.

# Extra positional arguments (required)
- `lb::AbstractVector{<:Real}`: lower bounds, length N (use -Inf for none)
- `ub::AbstractVector{<:Real}`: upper bounds, length N (use +Inf for none)

# Keyword Arguments
- `tol::Float64 = 1e-3`: error tolerance for step acceptance
- `adapt_tol::Bool = false`: dynamically adjust tolerance based on gradient norm
- `max_iter::Int = 100`: maximum number of accepted steps
- `grad_tol::Float64 = 1e-6`: convergence threshold on projected gradient norm
- `descent_tol::Float64 = 1e-14`: minimum |ΔC| to count as progress
- `descent_patience::Int = 4`: consecutive stagnant steps before stopping
- `climb_patience::Int = 10`: consecutive non-descending steps before stopping
- `verbosity::Int = 1`: 0 = silent, 1 = per-step, 2 = debug
- `record::Bool = false`: if true, return (cost, stepsize) trace as a matrix
- `error_norm_p::Real = 2`: p-norm for error estimation
"""
function adaptive_dopri5_box!(
        θs::AbstractVector{<:Number},
        h₀::Real,
        cost_fn::Function,
        lb::AbstractVector{<:Real},
        ub::AbstractVector{<:Real};
        kwargs...)

    N = length(θs)
    @assert length(lb) == N && length(ub) == N "lb, ub must have the same length as θs"
    @assert all(lb[i] ≤ ub[i] for i in 1:N)   "lb[i] ≤ ub[i] required for all i"
    @assert all(lb[i] ≤ θs[i] ≤ ub[i] for i in 1:N) "Initial θs must be feasible"
    @assert h₀ < 0 "Step size h₀ must be negative for gradient descent via imaginary time evolution"

    # --- Configuration --------------------------------------------------------
    tol              = Float64(get(kwargs, :tol, 1e-3))
    adapt_tol        = get(kwargs, :adapt_tol, false)::Bool
    tol_max_factor   = Float64(get(kwargs, :tol_max_factor, 5.0))  # cap: tol_cur ≤ tol_max_factor * tol
    ng_norm_max      = Float64(get(kwargs, :ng_norm_max, Inf))      # reject step & shrink h if ‖ng‖ exceeds this
    max_iter         = get(kwargs, :max_iter, 100)::Int
    grad_tol         = Float64(get(kwargs, :grad_tol, 1e-6))
    descent_tol      = Float64(get(kwargs, :descent_tol, 5e-8))
    descent_patience = get(kwargs, :descent_patience, 4)::Int
    climb_patience   = get(kwargs, :climb_patience, 15)::Int
    verbosity        = get(kwargs, :verbosity, 1)::Int
    record           = get(kwargs, :record, false)::Bool
    err_p            = Float64(get(kwargs, :error_norm_p, 2))
    ref_val          = get(kwargs, :ref_val, nothing)

    # --- DOPRI5 Butcher tableau (Dormand & Prince 1980) -----------------------
    a2 = (1/5,)
    a3 = (3/40,        9/40)
    a4 = (44/45,       -56/15,       32/9)
    a5 = (19372/6561,  -25360/2187,  64448/6561,  -212/729)
    a6 = (9017/3168,   -355/33,      46732/5247,   49/176,    -5103/18656)
    a7 = (35/384,       0.0,         500/1113,     125/192,   -2187/6784,   11/84)

    b̂ = (35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0)
    b  = (5179/57600, 0.0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40)
    ê  = ntuple(i -> b̂[i] - b[i], 7)

    # --- Box helpers ----------------------------------------------------------

    @inline function project_θ!(v)
        @inbounds @simd for i in 1:N
            v[i] = clamp(v[i], lb[i], ub[i])
        end
    end

    # Zero gradient components whose induced displacement h*g[i] would push
    # a coordinate further into an already-active bound.
    # With h < 0:
    #   lower bound active (v[i] ≈ lb[i]): displacement h*g[i] < 0 when g[i] > 0 → infeasible
    #   upper bound active (v[i] ≈ ub[i]): displacement h*g[i] > 0 when g[i] < 0 → infeasible
    @inline function project_grad!(g, v, h)
        @inbounds @simd for i in 1:N
            if (v[i] ≤ lb[i] && h * g[i] < 0) || (v[i] ≥ ub[i] && h * g[i] > 0)
                g[i] = 0.0
            end
        end
    end

    # --- Zygote wrapper -------------------------------------------------------
    function valgrad!(out_k, θ)
        result = Zygote.withgradient(cost_fn, θ)
        g = result.grad[1]
        if g === nothing
            fill!(out_k, 0.0)
        else
            copy!(out_k, g)
        end
        return result.val
    end

    # --- Pre-allocate workspace -----------------------------------------------
    k     = ntuple(_ -> Vector{Float64}(undef, N), 7)
    Δθ    = Vector{Float64}(undef, N)
    θ_new = Vector{Float64}(undef, N)

    # --- Stage evaluation -----------------------------------------------------
    # Standard tableau increment Δθ = θs + h * Σ w_i k_i.
    # h < 0 and k_i = ∇C, so each stage moves along −∇C.
    # The trial point is projected onto the box and the gradient is
    # cone-projected before being stored in out_k.
    @inline function eval_stage!(out_k, weights, stage_ks, h)
        copy!(Δθ, θs)
        @inbounds for (w, kᵢ) in zip(weights, stage_ks)
            iszero(w) && continue
            hw = h * w
            @simd for j in 1:N
                Δθ[j] += hw * kᵢ[j]
            end
        end
        project_θ!(Δθ)
        val = valgrad!(out_k, Δθ)
        project_grad!(out_k, Δθ, h)
        return val
    end

    # --- Error norm -----------------------------------------------------------
    @inline function error_norm(ê, ks_tuple, h, p)
        s = 0.0
        @inbounds for j in 1:N
            v = 0.0
            for i in 1:7
                iszero(ê[i]) && continue
                v += ê[i] * ks_tuple[i][j]
            end
            s += abs(h * v)^p
        end
        return s^(inv(p))
    end

    # --- State initialisation -------------------------------------------------
    h              = Float64(h₀)      # negative throughout
    tol_cur        = tol
    iter           = 0
    climb_count    = 0
    stagnate_count = 0

    project_θ!(θs)
    C₁     = valgrad!(k[1], θs)
    project_grad!(k[1], θs, h)
    prev_C = C₁
    ∇norm  = norm(k[1])

    trace = record ? [(cost=C₁, h=h)] : nothing

    verbosity ≥ 2 && @printf("Init: ‖∇C‖ = %4.4g, tol = %g\n", ∇norm, tol_cur)

    # --- Main loop (6 Zygote evaluations per accepted step via FSAL) ----------
    while iter ≤ max_iter && ∇norm ≥ grad_tol

        # Stages 2–6 (5 new evaluations)
        eval_stage!(k[2], a2, (k[1],), h)
        eval_stage!(k[3], a3, (k[1], k[2]), h)
        eval_stage!(k[4], a4, (k[1], k[2], k[3]), h)
        eval_stage!(k[5], a5, (k[1], k[2], k[3], k[4]), h)
        eval_stage!(k[6], a6, (k[1], k[2], k[3], k[4], k[5]), h)

        # Stage 7: evaluated at the 5th-order solution point (FSAL)
        C₇ = eval_stage!(k[7], a7, (k[1], k[2], k[3], k[4], k[5], k[6]), h)
        copy!(θ_new, Δθ)   # Δθ holds the projected stage-7 point

        err     = error_norm(ê, k, h, err_p)
        # exponent 0.2 = 1/5 for a 5th-order method; stretch keeps h negative
        stretch = clamp(0.9 * (tol_cur / err)^0.2, 0.5, 2.0)

        if err < tol_cur  # ---- accept ----------------------------------------
            copy!(θs, θ_new)       # θ_new is already feasible
            copy!(k[1], k[7])      # FSAL: k₇ (cone-projected) becomes k₁
            ∇norm = norm(k[1])

            iter += 1
            ΔC = C₇ - prev_C

            if verbosity ≥ 1 && isnothing(ref_val)
                @printf("iter %4d  C = %+.14e  ΔC = %+.3e  ‖∇C‖ = %.3e  h = %.4e  err = %.2e\n",
                        iter, C₇, ΔC, ∇norm, h, err)
            end
            if verbosity ≥ 1 && !isnothing(ref_val)
                @printf("iter %4d  C = %+.14e  ΔEᵣₑ = %.8e  ΔCᵢ = %+.3e  ‖∇C‖ = %.3e  h = %.4e  err = %.2e\n",
                        iter, C₇, C₇ - ref_val, ΔC, ∇norm, h, err)
            end

            record && push!(trace, (cost=C₇, h=h))

            if ΔC > 0
                climb_count += 1
                stagnate_count = 0
                verbosity ≥ 2 && @printf("  ⚠ non-descent ΔC = %+.4e (%d/%d)\n",
                                         ΔC, climb_count, climb_patience)
                if climb_count ≥ climb_patience
                    @info "Stopping: $climb_count non-descent steps"
                    break
                end
            elseif abs(ΔC) < descent_tol
                stagnate_count += 1
                climb_count = 0
                verbosity ≥ 2 && @printf("  ⚠ stagnation |ΔC| = %.3e (%d/%d)\n",
                                         abs(ΔC), stagnate_count, descent_patience)
                if stagnate_count ≥ descent_patience
                    @info "Stopping: cost stagnated ($stagnate_count steps, tol=$descent_tol)"
                    break
                end
            end

            prev_C = C₇
        else
            verbosity ≥ 2 && @warn "Step rejected (err=$(round(err; sigdigits=3))), shrinking h"
        end

        adapt_tol && (tol_cur = clamp(tol_cur * clamp(0.98 * (∇norm / grad_tol)^0.2, 0.95, 1.05),
                                      tol, tol_max_factor * tol))
        h *= stretch   # stretch > 0, so h stays negative
    end

    # --- Summary --------------------------------------------------------------
    if isnothing(ref_val)
        @printf("Done: %d iters, C = %+.14e, ‖∇C‖ = %.3e, h = %.4e\n",
            iter, prev_C, ∇norm, h)
    else
        @printf("Done: %d iters, C = %+.14e, ΔEᵣₑ = %.8e, ‖∇C‖ = %.3e, h = %.4e\n",
            iter, prev_C, prev_C - ref_val, ∇norm, h)
    end

    return record ? reduce(hcat, [[t.cost, t.h] for t in trace]) : nothing
end


"""
    adaptive_dopri5_ng_box!(θs, h₀, vgm_fn, p, lb, ub; kwargs...)

Adaptive DOPRI5 gradient-flow integrator using the **natural gradient** from
`value_gradient_and_metric` (or any compatible `vgm_fn`).

Identical to `adaptive_dopri5_box!` except the stage vectors `k[i]` are the
natural gradient `ng = g⁻¹·∇E` returned directly by `vgm_fn(θ, p)`, so no
Zygote evaluation is performed.  The metric tensor is recomputed at each stage
point (the metric changes along the trajectory).

## Arguments

- `θs`     — parameters, modified in-place
- `h₀`     — initial step size, must be **negative** (descent)
- `vgm_fn` — callable with signature `(θ, p; kwargs...) → (E, ng, g)`:
              energy, natural gradient, metric.  Typically one of the `vgm_*`
              wrappers with `natural_grad=true` (the default).
- `p`      — parameter tuple forwarded to `vgm_fn`
- `lb`,`ub`— box constraint vectors (same length as `θs`)

Keyword arguments are the same as `adaptive_dopri5_box!`:
`tol`, `max_iter`, `grad_tol`, `descent_tol`, `descent_patience`,
`climb_patience`, `verbosity`, `record`, `error_norm_p`, `ref_val`.
"""
function adaptive_dopri5_ng_box!(
        θs::AbstractVector{<:Number},
        h₀::Real,
        vgm_fn::Function,
        p,
        lb::AbstractVector{<:Real},
        ub::AbstractVector{<:Real};
        kwargs...)

    N = length(θs)
    @assert length(lb) == N && length(ub) == N "lb, ub must have the same length as θs"
    @assert all(lb[i] ≤ ub[i] for i in 1:N)   "lb[i] ≤ ub[i] required for all i"
    @assert all(lb[i] ≤ θs[i] ≤ ub[i] for i in 1:N) "Initial θs must be feasible"
    @assert h₀ < 0 "Step size h₀ must be negative for gradient descent via imaginary time evolution"

    # --- Configuration --------------------------------------------------------
    tol              = Float64(get(kwargs, :tol, 1e-5))
    adapt_tol        = get(kwargs, :adapt_tol, false)::Bool
    tol_max_factor   = Float64(get(kwargs, :tol_max_factor, 5.0))  # cap: tol_cur ≤ tol_max_factor * tol
    ng_norm_max      = Float64(get(kwargs, :ng_norm_max, Inf))      # reject step & shrink h if ‖ng‖ exceeds this
    max_iter         = get(kwargs, :max_iter, 100)::Int
    grad_tol         = Float64(get(kwargs, :grad_tol, 1e-6))
    descent_tol      = Float64(get(kwargs, :descent_tol, 5e-8))
    descent_patience = get(kwargs, :descent_patience, 4)::Int
    climb_patience   = get(kwargs, :climb_patience, 15)::Int
    verbosity        = get(kwargs, :verbosity, 1)::Int
    record           = get(kwargs, :record, false)::Bool
    err_p            = Float64(get(kwargs, :error_norm_p, 2))
    ref_val          = get(kwargs, :ref_val, nothing)

    # --- DOPRI5 Butcher tableau (Dormand & Prince 1980) -----------------------
    a2 = (1/5,)
    a3 = (3/40,        9/40)
    a4 = (44/45,       -56/15,       32/9)
    a5 = (19372/6561,  -25360/2187,  64448/6561,  -212/729)
    a6 = (9017/3168,   -355/33,      46732/5247,   49/176,    -5103/18656)
    a7 = (35/384,       0.0,         500/1113,     125/192,   -2187/6784,   11/84)

    b̂ = (35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0)
    b  = (5179/57600, 0.0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40)
    ê  = ntuple(i -> b̂[i] - b[i], 7)

    # --- Box helpers ----------------------------------------------------------
    @inline function project_θ!(v)
        @inbounds @simd for i in 1:N
            v[i] = clamp(v[i], lb[i], ub[i])
        end
    end

    @inline function project_grad!(g, v, h)
        @inbounds @simd for i in 1:N
            if (v[i] ≤ lb[i] && h * g[i] < 0) || (v[i] ≥ ub[i] && h * g[i] > 0)
                g[i] = 0.0
            end
        end
    end

    tol_floor_factor = Float64(get(kwargs, :tol_floor_factor, 1e-3)) # floor: tol_cur ≥ tol_floor_factor * tol
    tol_floor        = tol * tol_floor_factor

    # --- Natural gradient wrapper ---------------------------------------------
    # Calls vgm_fn, copies the natural gradient ng into out_k, returns energy E
    function valng!(out_k, θ)
        E, ng, _ = vgm_fn(θ, p; natural_grad=true)
        copy!(out_k, ng)
        return E
    end

    # --- Pre-allocate workspace -----------------------------------------------
    k     = ntuple(_ -> Vector{Float64}(undef, N), 7)
    Δθ    = Vector{Float64}(undef, N)
    θ_new = Vector{Float64}(undef, N)

    # --- Stage evaluation -----------------------------------------------------
    @inline function eval_stage!(out_k, weights, stage_ks, h)
        copy!(Δθ, θs)
        @inbounds for (w, kᵢ) in zip(weights, stage_ks)
            iszero(w) && continue
            hw = h * w
            @simd for j in 1:N
                Δθ[j] += hw * kᵢ[j]
            end
        end
        project_θ!(Δθ)
        val = valng!(out_k, Δθ)
        project_grad!(out_k, Δθ, h)
        return val
    end

    # --- Error norm -----------------------------------------------------------
    @inline function error_norm(ê, ks_tuple, h, p)
        s = 0.0
        @inbounds for j in 1:N
            v = 0.0
            for i in 1:7
                iszero(ê[i]) && continue
                v += ê[i] * ks_tuple[i][j]
            end
            s += abs(h * v)^p
        end
        return s^(inv(p))
    end

    # --- State initialisation -------------------------------------------------
    h              = Float64(h₀)
    tol_cur        = tol
    iter           = 0
    climb_count    = 0
    stagnate_count = 0

    project_θ!(θs)
    C₁     = valng!(k[1], θs)
    project_grad!(k[1], θs, h)
    prev_C = C₁
    ∇norm  = norm(k[1])

    E_best = C₁
    θ_best = copy(θs)

    trace = record ? [(cost=C₁, h=h)] : nothing

    verbosity ≥ 2 && @printf("Init: ‖ng‖ = %4.4g, tol = %g\n", ∇norm, tol_cur)

    # --- Main loop (6 vgm evaluations per accepted step via FSAL) -------------
    while iter ≤ max_iter && ∇norm ≥ grad_tol

        eval_stage!(k[2], a2, (k[1],), h)
        eval_stage!(k[3], a3, (k[1], k[2]), h)
        eval_stage!(k[4], a4, (k[1], k[2], k[3]), h)
        eval_stage!(k[5], a5, (k[1], k[2], k[3], k[4]), h)
        eval_stage!(k[6], a6, (k[1], k[2], k[3], k[4], k[5]), h)

        C₇ = eval_stage!(k[7], a7, (k[1], k[2], k[3], k[4], k[5], k[6]), h)
        copy!(θ_new, Δθ)

        err     = error_norm(ê, k, h, err_p)
        stretch = clamp(0.9 * (tol_cur / err)^0.2, 0.5, 2.0)

        if err < tol_cur  # ---- accept ----------------------------------------
            new_norm = norm(k[7])
            if new_norm > ng_norm_max   # ‖ng‖ explosion: treat as rejection
                verbosity ≥ 1 && @printf("  !! ||ng|| = %.3e > ng_norm_max, rejecting step\n", new_norm)
                stretch = 0.5           # aggressively shrink h
            else
                copy!(θs, θ_new)
                copy!(k[1], k[7])      # FSAL
                ∇norm = new_norm

                if C₇ < E_best
                    E_best = C₇
                    copy!(θ_best, θs)
                end

                iter += 1
                ΔC = C₇ - prev_C

                if verbosity ≥ 1 && isnothing(ref_val)
                    @printf("iter %4d  C = %+.14e  ΔC = %+.3e  ‖ng‖ = %.3e  h = %.4e  err = %.2e\n",
                            iter, C₇, ΔC, ∇norm, h, err)
                end
                if verbosity ≥ 1 && !isnothing(ref_val)
                    @printf("iter %4d  C = %+.14e  ΔEᵣₑ = %.8e  ΔCᵢ = %+.3e  ‖ng‖ = %.3e  h = %.4e  err = %.2e\n",
                            iter, C₇, C₇ - ref_val, ΔC, ∇norm, h, err)
                end

                record && push!(trace, (cost=C₇, h=h))

                if ΔC > 0
                    climb_count += 1
                    verbosity ≥ 2 && @printf("  ⚠ non-descent ΔC = %+.4e (%d/%d)\n",
                                             ΔC, climb_count, climb_patience)
                    if climb_count ≥ climb_patience
                        @info "Stopping: $climb_count total non-descent steps"
                        break
                    end
                end

                if abs(ΔC) < descent_tol
                    stagnate_count += 1
                    verbosity ≥ 2 && @printf("  ⚠ stagnation |ΔC| = %.3e (%d/%d)\n",
                                             abs(ΔC), stagnate_count, descent_patience)
                    if stagnate_count ≥ descent_patience
                        @info "Stopping: cost stagnated ($stagnate_count steps, tol=$descent_tol)"
                        break
                    end
                end

                prev_C = C₇
            end  # ng_norm_max guard
        else
            verbosity ≥ 2 && @warn "Step rejected (err=$(round(err; sigdigits=3))), shrinking h"
        end

        adapt_tol && (tol_cur = clamp(tol_cur * clamp(0.98 * (∇norm / grad_tol)^0.2, 0.95, 1.05),
                                      tol_floor, tol_max_factor * tol))
        h *= stretch
    end

    # --- Restore best-seen parameters ----------------------------------------
    if E_best < prev_C
        copy!(θs, θ_best)
        verbosity ≥ 1 && @printf("  ↩ restored best params (E_best = %.14e)\n", E_best)
        prev_C = E_best
    end

    # --- Summary --------------------------------------------------------------
    if isnothing(ref_val)
        @printf("Done: %d iters, C = %+.14e, ‖ng‖ = %.3e, h = %.4e\n",
            iter, prev_C, ∇norm, h)
    else
        @printf("Done: %d iters, C = %+.14e, ΔEᵣₑ = %.8e, ‖ng‖ = %.3e, h = %.4e\n",
            iter, prev_C, prev_C - ref_val, ∇norm, h)
    end

    return record ? reduce(hcat, [[t.cost, t.h] for t in trace]) : nothing
end


# ------------------------------------------------------------------------------
# Kitaev chain builders (AD-friendly)
# ------------------------------------------------------------------------------


function hamiltonian_variance(Γ, H_D, U)
    N = size(H_D, 1) ÷ 2
    ε = diag(H_D)[N+1:2N]
    ν = real(diag(U' * Γ * U))[1:N]
    var = 4 * mapreduce((e, n) -> e^2 * n * (1 - n), +, ε, ν; init=0.0)
    return max(var, 0.0)
end


function f_loss_NN_Rxx_Rz(x, p)
    # @assert iseven(length(x))
    ψ0, N, D, (hT, UT), output_flag, with_penalty = p
    ψ = copy(ψ0)
    T = eltype(x)

    # D = length(x)÷2

    for d in 1:D
        J = x[2*(d-1)+1]
        h = x[2*(d-1)+2]
        ψ = Evolve_exp_AD(ψ, TFI_Hamiltonian(N, J, 0.0; PBC=0.0), one(T))
        ψ = Evolve_exp_AD(ψ, TFI_Hamiltonian(N, 0.0, h; PBC=0.0), one(T))
    end

    energy = Fu.Energy(ψ, (hT, UT))
    loss = with_penalty ? energy + sqrt(hamiltonian_variance(ψ, hT, UT)) : energy
    return output_flag ? (loss, ψ) : loss
end


function f_loss_NN_TFI_Rz(x, p)
    # @assert iseven(length(x))
    ψ0, N, D, (hT, UT), output_flag, with_penalty = p
    ψ = copy(ψ0)
    T = eltype(x)

    # D = length(x)÷2

    nP = 3
    for d in 1:D
        tJ      = x[nP*(d-1)+1]
        hoverJ  = x[nP*(d-1)+2]
        angleZ  = x[nP*(d-1)+3]
        ψ = Evolve_exp_AD(ψ, TFI_Hamiltonian(N, tJ, hoverJ; PBC=0.0), one(T))
        ψ = Evolve_exp_AD(ψ, TFI_Hamiltonian(N, 0.0, angleZ; PBC=0.0), one(T))
    end

    energy = Fu.Energy(ψ, (hT, UT))
    loss = with_penalty ? energy + sqrt(hamiltonian_variance(ψ, hT, UT)) : energy
    return output_flag ? (loss, ψ) : loss
end


function f_loss_NN_TFI_only(x, p)
    # @assert iseven(length(x))
    ψ0, N, D, (hT, UT), output_flag, with_penalty = p
    ψ = copy(ψ0)
    T = eltype(x)

    # D = length(x)÷2

    nP = 2
    for d in 1:D
        tJ      = x[nP*(d-1)+1]
        hoverJ  = x[nP*(d-1)+2]
        # angleZ  = x[nP*(d-1)+3]
        # Hz = TFI_Hamiltonian(N, 0.0, angleZ; PBC=0.0)
        ψ = Evolve_exp_AD(ψ, TFI_Hamiltonian(N, tJ, hoverJ; PBC=0.0), one(T))
    end


    energy = Fu.Energy(ψ, (hT, UT))
    loss = with_penalty ? energy + sqrt(hamiltonian_variance(ψ, hT, UT)) : energy
    # loss = energy + sqrt(variance_H)
    return output_flag ? (loss, ψ) : loss
end


function f_loss_NN_Kitaev_only(x, p)
    # @assert iseven(length(x))
    ψ0, N, D, (hT, UT), output_flag, with_penalty = p
    ψ = copy(ψ0)
    T = eltype(x)

    # D = length(x)÷2

    nP = 3
    for d in 1:D
        tJ  = x[nP*(d-1)+1]
        ΔoverJ  = x[nP*(d-1)+2]
        μoverJ  = x[nP*(d-1)+3]
        ψ = Evolve_exp_AD(ψ, Kitaev_Hamiltonian(N, tJ, ΔoverJ, μoverJ; PBC=0.0), one(T))
    end


    energy = Fu.Energy(ψ, (hT, UT))
    loss = with_penalty ? energy + sqrt(hamiltonian_variance(ψ, hT, UT)) : energy
    # loss = energy + sqrt(variance_H)
    return output_flag ? (loss, ψ) : loss
end


function f_loss_NN_Kitaev_HVA(x, p)
    # @assert iseven(length(x))
    ψ0, N, D, (hT, UT), output_flag, with_penalty = p
    ψ = copy(ψ0)
    T = eltype(x)

    # D = length(x)÷2

    nP = 3
    for d in 1:D
        tJ  = x[nP*(d-1)+1]
        ΔoverJ  = x[nP*(d-1)+2]
        μoverJ  = x[nP*(d-1)+3]
        ψ = Evolve_exp_AD(
                Evolve_exp_AD(
                    Evolve_exp_AD(
                        ψ,
                        Kitaev_Hamiltonian(N, tJ, 0.0, 0.0; PBC=0.0),
                        one(T)
                    ),
                Kitaev_Hamiltonian(N, 0.0, ΔoverJ, 0.0; PBC=0.0),
                one(T)
                ),
            Kitaev_Hamiltonian(N, 0.0, 0.0, μoverJ; PBC=0.0),
            one(T)
        )
    end


    energy = Fu.Energy(ψ, (hT, UT))
    loss = with_penalty ? energy + sqrt(hamiltonian_variance(ψ, hT, UT)) : energy
    # loss = energy + sqrt(variance_H)
    return output_flag ? (loss, ψ) : loss
end


function f_loss_EXP_TFI_only(x, p)
    # @assert iseven(length(x))
    ψ0, N, D, (hT, UT), output_flag, with_penalty = p
    ψ = copy(ψ0)
    T = eltype(x)

    # D = length(x)÷2

    nP = 3
    for d in 1:D
        λ       = x[nP*(d-1)+1]
        tJ      = x[nP*(d-1)+2]
        hoverJ  = x[nP*(d-1)+3]
        # angleZ  = x[nP*(d-1)+4]
        ψ = Evolve_exp_AD(ψ, LongRangeTFI_exp_Hamiltonian(N, tJ, hoverJ, λ), one(T))
        # ψ = Evolve_exp_AD(ψ, TFI_Hamiltonian(N, 0.0, angleZ; PBC=0.0), one(T))
    end

    energy = Fu.Energy(ψ, (hT, UT))
    loss = with_penalty ? energy + sqrt(hamiltonian_variance(ψ, hT, UT)) : energy
    return output_flag ? (loss, ψ) : loss
end


function f_loss_POW_TFI_only(x, p)
    # @assert iseven(length(x))
    ψ0, N, D, (hT, UT), output_flag, with_penalty = p
    ψ = copy(ψ0)
    T = eltype(x)

    # D = length(x)÷2

    nP = 3
    for d in 1:D
        α       = x[nP*(d-1)+1]
        tJ      = x[nP*(d-1)+2]
        hoverJ  = x[nP*(d-1)+3]
        ψ = Evolve_exp_AD(ψ, LongRangeTFI_pow_Hamiltonian(N, tJ, hoverJ, α), one(T))
        # ψ = Evolve_exp_AD(ψ, Hz, one(T))
    end

    energy = Fu.Energy(ψ, (hT, UT))
loss = with_penalty ? energy + sqrt(hamiltonian_variance(ψ, hT, UT)) : energy
    return output_flag ? (loss, ψ) : loss
end


function f_loss_NN_TFI_brickwall(x, p)
    # @assert iseven(length(x))
    ψ0, N, D, (hT, UT), output_flag, with_penalty = p
    ψ = copy(ψ0)
    T = eltype(x)

    # D = length(x)÷2


    for d in 1:D
        tJ1      = x[4*(d-1)+1]
        hoverJ1  = x[4*(d-1)+2]
        tJ2      = x[4*(d-1)+3]
        hoverJ2  = x[4*(d-1)+4]

        tJ1_vec = repeat([tJ1,zero(T)],N÷2)[1:N-1]
        tJ2_vec = repeat([zero(T),tJ2],N÷2)[1:N-1]

        ψ = Evolve_exp_AD(ψ,
            TFI_Hamiltonian(N, tJ1_vec, fill(hoverJ1, N); PBC=0.0),
            one(T)
        )
        ψ = Evolve_exp_AD(ψ,
            # HTFI2,
            TFI_Hamiltonian(N, tJ2_vec, fill(hoverJ2, N); PBC=0.0),
            one(T),
        )
    end

    energy = Fu.Energy(ψ, (hT, UT))
    loss = with_penalty ? energy + sqrt(hamiltonian_variance(ψ, hT, UT)) : energy
    return output_flag ? (loss, ψ) : loss
end


function ES_spec_ratios(Gamma_psi; mode_cutoff=10, normalisation_factor = 1.0)
    N = size(Gamma_psi, 1) ÷ 2

    Gamma_psi_A = if N == 2
        Fu.Reduce_gamma(Gamma_psi, N ÷ 2, 1)
    elseif iseven(N ÷ 2)
        Fu.Reduce_gamma(Gamma_psi, N ÷ 2, 1)
    else
        Fu.Reduce_gamma(Gamma_psi, N ÷ 2 + 1, 1)
    end
    eigs_even, eigs_odd = Fu.approx_eigenvalues_of_rho(Gamma_psi_A; mode_cutoff)
    spec_even = -log.(abs.(eigs_even))
    spec_odd = -log.(abs.(eigs_odd))
    gaps_even = spec_even .- spec_even[1]
    gaps_odd = spec_odd .- spec_even[1]
    ratios_even = normalisation_factor * gaps_even ./ gaps_even[2]
    ratios_odd = normalisation_factor * gaps_odd ./ gaps_even[2]
    return (ratios_even, ratios_odd)
end


function ES_spec_ratios(eigs_even, eigs_odd; normalisation_factor = 1.0)
    spec_even = -log.(abs.(eigs_even))
    spec_odd = -log.(abs.(eigs_odd))
    gaps_even = spec_even .- spec_even[1]
    gaps_odd = spec_odd .- spec_even[1]
    ratios_even = normalisation_factor * gaps_even ./ gaps_even[2]
    ratios_odd = normalisation_factor * gaps_odd ./ gaps_even[2]
    return (ratios_even, ratios_odd)
end


function ES_spec(Gamma_psi; mode_cutoff=10, normalisation_factor = 1.0)
    N = size(Gamma_psi, 1) ÷ 2

    Gamma_psi_A = if N == 2
        Fu.Reduce_gamma(Gamma_psi, N ÷ 2, 1)
    elseif iseven(N ÷ 2)
        Fu.Reduce_gamma(Gamma_psi, N ÷ 2, 1)
    else
        Fu.Reduce_gamma(Gamma_psi, N ÷ 2 + 1, 1)
    end
    eigs_even, eigs_odd = Fu.approx_eigenvalues_of_rho(Gamma_psi_A; mode_cutoff)
    return (eigs_even, eigs_odd)
end


psi_up(L::Int) = Hermitian(
    [
        diagm(ones(ComplexF64, L)) zeros(ComplexF64, L, L);
        zeros(ComplexF64, L, L) zeros(ComplexF64, L, L)
    ]
)


function Build_A_TFI(N::Int, J::Number, h::Number, PBC::Number)
    diag = 2h * ones(N)
    one_off = J * ones(N - 1)
    return -0.5 * LinearAlgebra.diagm(
        -1 => one_off,
        0 => diag,
        1 => one_off,
    ) + diagm(
        N - 1 => [-0.5 * PBC * J],
        -(N - 1) => [-0.5 * PBC * J]
    )
end


function Build_B_TFI(N::Int, J::Number, PBC::Number)
    hopping = J * ones(N - 1)
    return -0.5 * LinearAlgebra.diagm(
        1 => -hopping,
        -1 => hopping,
    ) + diagm(
        N - 1 => [-0.5 * PBC * J],
        -(N - 1) => [0.5 * PBC * J]
    )
    return
end


function Build_A_TFI(N, J::AbstractVector{T1}, h::AbstractVector{T2}, PBC::Number) where T1  where T2
    @assert length(h) == N
    @assert length(J) == N - 1
    return -0.5 * LinearAlgebra.diagm(
        -1 => J,
        0 => 2h,
        1 => J,
    ) + diagm(
        N - 1 => [-PBC],
        -(N - 1) => [-PBC]
    )
end


function Build_B_TFI(N, J::AbstractVector{<:Number}, PBC::Number)
    @assert length(J) == N - 1
    return -0.5 * LinearAlgebra.diagm(
        -1 => J,
        # 0 => zeros(Float64, N),
        1 => -J,
    ) + diagm(
        N - 1 => [-PBC],
        -(N - 1) => [PBC]
    )
    return
end


function TFI_Hamiltonian(N, J, h; PBC=0.0)
    A = Build_A_TFI(N, J, h, PBC)
    B = Build_B_TFI(N, J, PBC)
    return Hermitian([-A B; -B A]) ## note that `Hermitian` is AD-friendly while `Symmetric` is not!!
end


function Kitaev_Hamiltonian(N, Ja, Jb, h; PBC=0.0)
    A = Build_A_TFI(N, Ja, h, PBC)
    B = Build_B_TFI(N, Jb, PBC)
    return Hermitian([-A B; -B A]) ## note that `Hermitian` is AD-friendly while `Symmetric` is not!!
end


"""    _pair_distances(N) → Vector{Int}

Precompute distances |j - i| for all pairs (i,j) with j > i, in the order
(1,2), (1,3), …, (1,N), (2,3), …, (N-1,N). Length = N(N-1)/2.
"""
_pair_distances(N::Int) = [j - i for i in 1:N for j in (i+1):N]


"""
    _pair_idx(i, d, N) → Int

Linear index into the upper-triangular coupling vector for pair (i, i+d).
"""
@inline _pair_idx(i, d, N) = (i - 1) * N - i * (i - 1) ÷ 2 + d



#  Coupling-vector builders                                                   
#                                                                              
#  Uses `map` over a precomputed Int vector — Zygote-safe because:             
#   - _pair_distances(N) is pure integers, no tracked values                  
#   - map(f, ::Vector{Int}) has a proper Zygote adjoint                       
#   - the closure f captures only the differentiable parameters (J, λ, α)     


build_exp_couplings(N::Int, J::Number, λ::Number) =
    reduce(vcat, map(i -> map(d -> J * λ^(d - 1), 1:N-i), 1:N-1))


# ∂J_{ij}/∂λ = J·(d-1)·λ^{d-2}  (zero for d=1, regular for all λ including λ=0)
build_dexp_dlambda(N::Int, J::Number, λ::Number) =
    reduce(vcat, map(i -> map(d -> J * (d - 1) * (d >= 2 ? λ^(d - 2) : zero(λ)), 1:N-i), 1:N-1))


build_pow_couplings(N::Int, J::Number, α::Number) =
    reduce(vcat, map(i -> map(d -> J / d^α, 1:N-i), 1:N-1))


# ∂J_{ij}/∂α = -J·log(d)/d^α  (zero for d=1 since log(1)=0)
build_dpow_dalpha(N::Int, J::Number, α::Number) =
    reduce(vcat, map(i -> map(d -> -J * log(Float64(d)) / d^α, 1:N-i), 1:N-1))



function Build_A_LongRangeTFI(N, J_upper::AbstractVector{T1},
                              h::AbstractVector{T2}) where {T1, T2}
    @assert length(h) == N
    @assert length(J_upper) == N * (N - 1) ÷ 2
    _Jd(d) = [J_upper[_pair_idx(i, d, N)] for i in 1:N-d]
    return -0.5 * (
        diagm(0 => 2h) + sum(
            diagm(d => _Jd(d), -d => _Jd(d))
            for d in 1:N-1
        )
    )
end


function Build_B_LongRangeTFI(N, J_upper::AbstractVector{<:Number})
    @assert length(J_upper) == N * (N - 1) ÷ 2
    _Jd(d) = [J_upper[_pair_idx(i, d, N)] for i in 1:N-d]
    return -0.5 * sum(
        diagm(d => -_Jd(d), -d => _Jd(d))
        for d in 1:N-1
    )
end


# 
#   Hamiltonians
# 

"""
    LongRangeTFI_Hamiltonian(N, J_upper, h_vec)

Universal entry point: builds the 2N×2N Hamiltonian from an arbitrary
upper-triangular coupling vector `J_upper` (length N(N-1)/2) and a
site-dependent field vector `h_vec` (length N).

Zygote-friendly, OBC only.
"""
function LongRangeTFI_Hamiltonian(N, J_upper::AbstractVector, h::AbstractVector)
    @assert length(J_upper) == N * (N - 1) ÷ 2
    @assert length(h) == N
    A = Build_A_LongRangeTFI(N, J_upper, h)
    B = Build_B_LongRangeTFI(N, J_upper)
    return Hermitian([-A B; -B A])
end


#
# Scalar convenience constructors
# 

"""
    LongRangeTFI_exp_Hamiltonian(N, J, h, λ)

Exponentially decaying long-range TFI. Reduces to standard TFI at λ = 0.
"""
function LongRangeTFI_exp_Hamiltonian(N::Int, J::Number, h::Number, λ::Number)
    return LongRangeTFI_Hamiltonian(N,
        build_exp_couplings(N, J, λ),
        h * ones(N),
    )
end


"""
    LongRangeTFI_pow_Hamiltonian(N, J, h, α)

Power-law decaying long-range TFI. Reduces to standard TFI as α → ∞.
"""
function LongRangeTFI_pow_Hamiltonian(N::Int, J::Number, h::Number, α::Number)
    return LongRangeTFI_Hamiltonian(N,
        build_pow_couplings(N, J, α),
        h * ones(N),
    )
end


# ==============================================================================
# Tridiagonal Fubini-Study metric  (BandedMatrices bandwidth-1)
# ==============================================================================
#
# For a free-fermion variational circuit applied as a sequence of Gaussian
# unitaries U_s = exp(2i·θ_s·G_s), the quantum geometric tensor is
#
#   Q_{kl} = ⟨∂_k ψ|(1-|ψ⟩⟨ψ|)|∂_l ψ⟩
#
# with tangent vectors  |∂_k ψ_D⟩ = 2i·G̃_k·|ψ_D⟩  where G̃_k is the generator
# G_k "dressed" by all subsequent circuit layers (Heisenberg picture).
#
# Key identity (cancellation of forward unitaries):
#   g_{k,k} = 4·Var_{Γ_k}(G_k)          — state AFTER stage k, NO dressing
#   g_{k,l} (same stage) = 4·Cov_{Γ_d}(G_k, G_l)   — NO dressing
#   g_{k,l} (adjacent stages d,d+1) = 4·Cov_{Γ_d}(G_k, Ũ_{d+1}†·G_l·U_{d+1})
#                                                     — ONE-STEP backward dress
#
# Via the BdG Wick theorem for Gaussian state Γ:
#   Cov_Γ(A,B) = Re[Tr(A·Γ·B·(I-Γ))]   (particle-hole factor 2 already included)
#
# Cost: O(S·N³) — same as the gradient forward pass.  No matrix exponentials
# in the backward direction; stored U_s matrices are reused for dressing.
#
# For combined exponentials exp(2i·H) where H = Σ_k θ_k·G_k (non-separable),
# the exact Fréchet formula is replaced by the linearisation G_k = ∂H/∂θ_k
# evaluated at the post-stage state Γ_d.  This is exact when generators
# commute and a good first-order approximation otherwise.
# ==============================================================================

"""
    _symmetrize_hermitian!(A)

In-place Hermitian symmetrization: `A ← (A + A') / 2`, without allocating.
"""
@inline function _symmetrize_hermitian!(A)
    M = size(A, 1)
    @inbounds for j in 1:M
        A[j, j] = real(A[j, j]) + 0im
        for i in 1:j-1
            v = (A[i, j] + conj(A[j, i])) * 0.5
            A[i, j] = v
            A[j, i] = conj(v)
        end
    end
    return A
end


"""
    _bdg_wick_cov!(tmp, W, B, Γ) → Float64

`Re[Tr(W·B·(I-Γ))]` where `W = A·Γ` (pre-computed) — BdG Wick connected
correlator `Cov_Γ(A, B)`.  Uses `Re[Tr(W·B)] - Re[dot(Γ, W·B)]` (the second
term equals `Re[Tr(Γ·W·B)]` for Hermitian Γ via Julia's `dot` identity).
`tmp` is a 2N×2N scratch buffer overwritten on exit.
"""
@inline function _bdg_wick_cov!(tmp, W, B, Γ)
    mul!(tmp, W, B)                           # tmp = W·B = A·Γ·B
    return real(tr(tmp)) - real(dot(Γ, tmp))  # Re[Tr(tmp·(I-Γ))]
end


"""
    _fréchet_phi!(Φ, d, E)

Fill `Φ[m,n]` in-place with the divided-difference weights for the Fréchet
derivative of `exp(2i·H)` in the eigenbasis of H, where `d[k] = H[k,k]` and
`E[k] = exp(2i·d[k])`:

    Φ[m,n] = (E[m] − E[n]) / (2i·(d[m]−d[n]))   for |d[m]−d[n]| > ε
    Φ[m,n] = E[m]                                  otherwise (L'Hôpital limit)

Together with the eigenvectors `VH` of H, the exact Fréchet gradient is

    ∂U/∂θ_k = 2i · VH · (Φ ⊙ (VH'·G_k·VH)) · VH'

which replaces the approximation `∂U/∂θ_k ≈ 2i·G_k·U` used when [H,G_k]≠0.
"""
function _fréchet_phi!(Φ, d::AbstractVector{Float64}, E::AbstractVector{ComplexF64})
    M = length(d)
    @inbounds for n in 1:M, m in 1:M
        Δ = d[m] - d[n]
        Φ[m,n] = abs(Δ) < 1e-12 ? E[m] : (E[m] - E[n]) / (2im * Δ)
    end
    return Φ
end


"""
    value_gradient_and_metric(ψ0, circuit, hT, UT; regularise=1e-10)

Energy value, adjoint gradient, and tridiagonal Fubini-Study metric tensor for a
free-fermion variational circuit.  All three quantities share a single forward
pass (O(S·N³)), making this cheaper than computing them separately.

## Circuit specification

`circuit` is a `Vector` of stage descriptors.  Each stage descriptor is either:

- `(G, θ)::Tuple{Matrix,Real}` — **separable stage**: one parameter, one generator.
  Unitary = `exp(2i·θ·G)`.  Gradient is exact.

- `(Gs, θs, (D_H, VH))::Tuple{Vector,Vector,Tuple}` — **combined stage with
  eigendecomposition**: `D_H, VH = Fu.Diag_h(H)` where `H = Σₖ θs[k]·Gs[k]`.
  Evolution uses `Diagonal(exp.(2i·diag(D_H)))` (no `exp` of a dense matrix).
  Gradient uses the exact Fréchet derivative via divided differences — no BCH
  approximation regardless of whether [H, Gₖ] = 0.

## Returns

- `E::Float64` — energy expectation value
- second element — depends on `natural_grad` (default `true`):
  - `true`:  natural gradient (solves `g*x = grad_E` via `Symmetric(g) \\ grad_E`)
  - `false`: plain gradient `∇E::Vector{Float64}`
- `g` — metric tensor, P×P.  Type depends on `full_metric`:
  - `false` (default): `BandedMatrix{Float64}`, bandwidth (1,1), tridiagonal approximation
  - `true`: `Matrix{Float64}`, full Fubini-Study metric, cost O(P²·N³)

## Metric entries

For parameters k (stage sₖ) and l (stage sₗ ≥ sₖ):

    g[k,l] = 2·Re[Tr(Gₖ·Γₛₖ·G̃ₗ→ₛₖ·(I−Γₛₖ))]

where `G̃ₗ→ₛₖ = Uₛₖ₊₁'·…·Uₛₗ'·Gₗ·Uₛₗ·…·Uₛₖ₊₁` is Gₗ dressed backward to stage sₖ.

With `full_metric=false` only the diagonal and adjacent-parameter entries are computed
(one-step backward dressing); all others are set to zero.

## Full metric algorithm

Column sweep: for each column l (left to right), maintain one dressed copy of Gₗ and
walk it backward stage by stage, reusing the dressed version for all k < l at the same
or earlier stages.  Precomputed `Wₖ = Gₖ·Γₛₖ` removes redundant multiplications.
Total cost: O(P·S·N³) dressing + O(P²·N³) Wick evaluations ≈ O(P²·N³).
"""
function value_gradient_and_metric(
        ψ0,
        circuit::AbstractVector,
        hT, UT;
        regularise::Float64 = 1e-10,
        full_metric::Bool   = true,
        natural_grad::Bool  = true,
        return_state::Bool  = false,
    )

    M2N = size(ψ0, 1)   # = 2N
    N   = M2N ÷ 2
    D   = length(circuit)

    # ── Pre-count total parameters P ──────────────────────────────────────────
    P = 0
    for stage in circuit
        P += stage isa Tuple{<:AbstractMatrix, <:Real} ? 1 : length(stage[1])
    end

    # ── Pre-allocate per-stage and per-parameter storage ──────────────────────
    Γ_per_stage  = [Matrix{ComplexF64}(undef, M2N, M2N) for _ in 1:D]
    Γ_prev_stage = [Matrix{ComplexF64}(undef, M2N, M2N) for _ in 1:D]
    U_per_stage  = Vector{Matrix{ComplexF64}}(undef, D)
    gen_list     = Vector{Matrix{ComplexF64}}(undef, P)
    stage_idx    = Vector{Int}(undef, P)

    # ── Workspace matrices (reused throughout) ─────────────────────────────────
    Γ_cur = Matrix{ComplexF64}(ψ0)
    tmp1  = Matrix{ComplexF64}(undef, M2N, M2N)
    tmp2  = Matrix{ComplexF64}(undef, M2N, M2N)
    W     = Matrix{ComplexF64}(undef, M2N, M2N)   # metric: G_k·Γ_s; backward: Λ̃
    G̃buf  = Matrix{ComplexF64}(undef, M2N, M2N)   # metric: dressed Gₗ; backward: Γ̃_prev
    Φbuf  = Matrix{ComplexF64}(undef, M2N, M2N)   # Fréchet divided-difference matrix

    # ── Forward pass ──────────────────────────────────────────────────────────
    k = 0
    for s in 1:D
        stage = circuit[s]
        copyto!(Γ_prev_stage[s], Γ_cur)   # save state before this stage

        if stage isa Tuple{<:AbstractMatrix, <:Real}
            # ── Separable stage (G, θ) ─────────────────────────────────────
            G, θ = stage
            U = exp(2im * Float64(θ) .* Matrix{ComplexF64}(G))
            mul!(tmp1, U, Γ_cur)
            mul!(tmp2, tmp1, U')
            _symmetrize_hermitian!(tmp2)
            copyto!(Γ_per_stage[s], tmp2)
            U_per_stage[s] = U
            copyto!(Γ_cur, tmp2)
            k += 1
            gen_list[k]  = G isa Matrix{ComplexF64} ? G : Matrix{ComplexF64}(G)
            stage_idx[k] = s
        else
            # ── Combined stage (Gs, θs, (D_H, VH)) ────────────────────────
            Gs, _, (D_H, VH) = stage
            # Form U = VH · Diagonal(exp(2i·d)) · VH'  (O(N²) + N³)
            # then reuse it for Γ_new = U·Γ·U'  (2 N³) — 3 N³ total
            # vs. eigenbasis roundtrip + separate U build = 5 N³
            d = real.(diag(D_H))
            mul!(tmp1, VH, Diagonal(exp.(2im .* d)))     # O(N²): scales columns of VH
            mul!(tmp2, tmp1, VH')                        # N³: U = VH·E·VH'
            U_per_stage[s] = copy(tmp2)                  # store U
            mul!(tmp1, tmp2, Γ_cur)                      # N³: U·Γ
            mul!(tmp2, tmp1, U_per_stage[s]')            # N³: U·Γ·U'
            _symmetrize_hermitian!(tmp2)
            copyto!(Γ_per_stage[s], tmp2)
            copyto!(Γ_cur, tmp2)
            for Gk in Gs
                k += 1
                gen_list[k]  = Gk isa Matrix{ComplexF64} ? Gk : Matrix{ComplexF64}(Gk)
                stage_idx[k] = s
            end
        end
    end

    # ── Energy value ──────────────────────────────────────────────────────────
    B_mat = zeros(Float64, M2N, M2N)
    @inbounds for i in 1:N
        B_mat[i,   i  ] = real(hT[i+N, i+N])
        B_mat[i+N, i+N] = real(hT[i,   i  ])
    end
    mul!(tmp1, UT, B_mat)
    mul!(tmp2, tmp1, UT')
    _symmetrize_hermitian!(tmp2)
    Λ_init = copy(tmp2)
    E = real(dot(Λ_init, Γ_cur))

    # ── Metric ─────────────────────────────────────────────────────────────────
    if !full_metric
        # ── Tridiagonal approximation (bandwidth 1) ────────────────────────
        g = BandedMatrix(Zeros(P, P), (1, 1))

        for k in 1:P
            s  = stage_idx[k]
            Γs = Γ_per_stage[s]
            Gk = gen_list[k]
            mul!(W, Gk, Γs)
            g[k, k] = 2 * _bdg_wick_cov!(tmp1, W, Gk, Γs) + regularise

            k < P || continue
            sk1 = stage_idx[k+1]
            Gl  = gen_list[k+1]
            if sk1 == s
                g[k, k+1] = 2 * _bdg_wick_cov!(tmp1, W, Gl, Γs)
            else
                Us1 = U_per_stage[sk1]
                mul!(tmp1, Us1', Gl)
                mul!(G̃buf, tmp1, Us1)
                g[k, k+1] = 2 * _bdg_wick_cov!(tmp1, W, G̃buf, Γs)
            end
            g[k+1, k] = g[k, k+1]
        end

    else
        # ── Full P×P metric via column sweep ───────────────────────────────
        # Precompute W_list[k] = Gₖ·Γₛₖ  (one per parameter, shared across columns)
        W_list = Vector{Matrix{ComplexF64}}(undef, P)
        for k in 1:P
            W_list[k] = Matrix{ComplexF64}(undef, M2N, M2N)
            mul!(W_list[k], gen_list[k], Γ_per_stage[stage_idx[k]])
        end

        g      = zeros(Float64, P, P)
        G̃_col  = Matrix{ComplexF64}(undef, M2N, M2N)   # dressed column generator

        for l in 1:P
            sl  = stage_idx[l]
            Γsl = Γ_per_stage[sl]
            # Diagonal
            g[l, l] = 2 * _bdg_wick_cov!(tmp1, W_list[l], gen_list[l], Γsl)

            # Off-diagonal: walk k from l-1 down to 1, dressing G̃_col leftward
            copyto!(G̃_col, gen_list[l])
            cur_s = sl

            for k in l-1:-1:1
                s_k = stage_idx[k]
                # Dress G̃_col backward through stages until reaching stage s_k
                while cur_s > s_k
                    Us = U_per_stage[cur_s]
                    mul!(tmp1, Us', G̃_col)
                    mul!(tmp2, tmp1, Us)
                    copyto!(G̃_col, tmp2)
                    cur_s -= 1
                end
                val = 2 * _bdg_wick_cov!(tmp1, W_list[k], G̃_col, Γ_per_stage[s_k])
                g[k, l] = val
                g[l, k] = val
            end
        end

        # Regularise diagonal
        @inbounds for k in 1:P
            g[k, k] += regularise
        end

    end

    # ── Adjoint gradient (backward pass) ──────────────────────────────────────
    Λ_cur  = copy(Λ_init)
    ∇E     = zeros(Float64, P)
    k_flat = P

    for s in D:-1:1
        stage = circuit[s]
        Us    = U_per_stage[s]

        if stage isa Tuple{<:AbstractMatrix, <:Real}
            # ── Separable stage: exact via [G_k, Γ_s] ─────────────────────
            Gk = gen_list[k_flat]
            Γs = Γ_per_stage[s]
            mul!(tmp1, Gk, Γs)
            mul!(tmp2, Γs, Gk)
            tmp1 .-= tmp2                      # [G_k, Γ_s]
            mul!(tmp2, Λ_cur, tmp1)
            ∇E[k_flat] = -2 * imag(tr(tmp2))
            k_flat -= 1
        else
            # ── Combined stage: exact Fréchet gradient ─────────────────────
            # ∇E[k] = -4 Im( Σᵢ conj(E[i]) · [Λ̃·(Φ⊙G̃_k)·Γ̃_{s-1}][i,i] )
            Gs, _, (D_H, VH) = stage
            d     = real.(diag(D_H))
            E_vec = exp.(2im .* d)
            _fréchet_phi!(Φbuf, d, E_vec)

            # Transform Λ and Γ_{s-1} to H-eigenbasis (shared across all k in stage)
            # W    ← Λ̃ = VH'·Λ_cur·VH
            # G̃buf ← Γ̃_prev = VH'·Γ_{s-1}·VH
            mul!(tmp1, VH', Λ_cur);       mul!(W,    tmp1, VH)
            mul!(tmp1, VH', Γ_prev_stage[s]); mul!(G̃buf, tmp1, VH)

            for _ in length(Gs):-1:1
                Gk = gen_list[k_flat]
                # G̃_k = VH'·G_k·VH, then Φ ⊙ G̃_k (in tmp2)
                mul!(tmp1, VH', Gk);  mul!(tmp2, tmp1, VH)   # tmp2 = G̃_k
                tmp2 .*= Φbuf                                 # tmp2 = Φ ⊙ G̃_k
                # M = Λ̃·(Φ⊙G̃_k)·Γ̃_prev  (W = Λ̃, G̃buf = Γ̃_prev)
                mul!(tmp1, W, tmp2)
                mul!(tmp2, tmp1, G̃buf)
                # ∇E[k] = -4 Im(Tr(E⁻¹·M)) = -4 Im(Σᵢ conj(E[i])·M[i,i])
                # dot(E_vec, v) = Σᵢ conj(E_vec[i])·v[i] = Σᵢ conj(E[i])·M[i,i]
                ∇E[k_flat] = -4 * imag(dot(E_vec, diag(tmp2)))
                k_flat -= 1
            end
        end

        # Pull co-state: Λ ← U_s'·Λ·U_s
        mul!(tmp1, Us', Λ_cur)
        mul!(tmp2, tmp1, Us)
        _symmetrize_hermitian!(tmp2)
        copyto!(Λ_cur, tmp2)
    end
    if natural_grad
        F    = eigen(Symmetric(g))
        λmax = maximum(F.values)
        λmin = regularise * max(λmax, 1.0)   # relative cutoff: drop directions < regularise × λ_max
        λreg = max.(F.values, λmin)
        ng   = F.vectors * (F.vectors' * ∇E ./ λreg)
        return_state ? (return E, ng, g, Γ_cur) : (return E, ng, g)
    else
        return_state ? (return E, ∇E, g, Γ_cur) : (return E, ∇E, g)
    end
end


"""
    value_and_gradient(ψ0, circuit, hT, UT; return_state=false)

Energy value and exact adjoint gradient for a free-fermion variational circuit,
without computing the Fubini–Study metric.  Stripped-down counterpart of
`value_gradient_and_metric` for use with optimisers that do not need the QFI
metric (plain gradient descent, L-BFGS, conjugate gradient, Optim.jl).

## Cost
- Forward sweep:  O(D·N³)
- Backward sweep: O(D·N³)
- Total:          O(D·N³)   (vs. O(P²·N³) for the full metric variant)
- Memory:         O(D·N²)   (only per-stage Γ, U, Γ_prev)

## Circuit specification
Identical to `value_gradient_and_metric`. Each stage descriptor is either:

- `(G, θ)::Tuple{Matrix,Real}` — separable stage with unitary `exp(2i·θ·G)`.
- `(Gs, θs, (D_H, VH))::Tuple{Vector,Vector,Tuple}` — combined stage with
  `H = Σₖ θs[k]·Gs[k]` and `D_H, VH = Fu.Diag_h(H)`.  Gradient uses the exact
  Fréchet derivative via divided differences.

## Returns
- `E::Float64` — energy expectation value
- `∇E::Vector{Float64}` — plain gradient (length P, total parameter count)
- `Γ_cur::Matrix{ComplexF64}` — final covariance, only if `return_state=true`
"""
function value_and_gradient(
        ψ0,
        circuit::AbstractVector,
        hT, UT;
        return_state::Bool = false,
    )

    M2N = size(ψ0, 1)   # = 2N
    N   = M2N ÷ 2
    D   = length(circuit)

    # ── Pre-count total parameters P ──────────────────────────────────────────
    P = 0
    for stage in circuit
        P += stage isa Tuple{<:AbstractMatrix, <:Real} ? 1 : length(stage[1])
    end

    # ── Pre-allocate per-stage storage ────────────────────────────────────────
    Γ_per_stage  = [Matrix{ComplexF64}(undef, M2N, M2N) for _ in 1:D]
    Γ_prev_stage = [Matrix{ComplexF64}(undef, M2N, M2N) for _ in 1:D]
    U_per_stage  = Vector{Matrix{ComplexF64}}(undef, D)
    gen_list     = Vector{Matrix{ComplexF64}}(undef, P)

    # ── Workspace matrices (reused throughout) ─────────────────────────────────
    Γ_cur = Matrix{ComplexF64}(ψ0)
    tmp1  = Matrix{ComplexF64}(undef, M2N, M2N)
    tmp2  = Matrix{ComplexF64}(undef, M2N, M2N)
    W     = Matrix{ComplexF64}(undef, M2N, M2N)   # backward: Λ̃ in H-eigenbasis
    G̃buf  = Matrix{ComplexF64}(undef, M2N, M2N)   # backward: Γ̃_prev in H-eigenbasis
    Φbuf  = Matrix{ComplexF64}(undef, M2N, M2N)   # Fréchet divided-difference matrix

    # ── Forward pass ──────────────────────────────────────────────────────────
    k = 0
    for s in 1:D
        stage = circuit[s]
        copyto!(Γ_prev_stage[s], Γ_cur)

        if stage isa Tuple{<:AbstractMatrix, <:Real}
            G, θ = stage
            U = exp(2im * Float64(θ) .* Matrix{ComplexF64}(G))
            mul!(tmp1, U, Γ_cur)
            mul!(tmp2, tmp1, U')
            _symmetrize_hermitian!(tmp2)
            copyto!(Γ_per_stage[s], tmp2)
            U_per_stage[s] = U
            copyto!(Γ_cur, tmp2)
            k += 1
            gen_list[k] = G isa Matrix{ComplexF64} ? G : Matrix{ComplexF64}(G)
        else
            Gs, _, (D_H, VH) = stage
            d = real.(diag(D_H))
            mul!(tmp1, VH, Diagonal(exp.(2im .* d)))
            mul!(tmp2, tmp1, VH')
            U_per_stage[s] = copy(tmp2)
            mul!(tmp1, tmp2, Γ_cur)
            mul!(tmp2, tmp1, U_per_stage[s]')
            _symmetrize_hermitian!(tmp2)
            copyto!(Γ_per_stage[s], tmp2)
            copyto!(Γ_cur, tmp2)
            for Gk in Gs
                k += 1
                gen_list[k] = Gk isa Matrix{ComplexF64} ? Gk : Matrix{ComplexF64}(Gk)
            end
        end
    end

    # ── Energy value ──────────────────────────────────────────────────────────
    B_mat = zeros(Float64, M2N, M2N)
    @inbounds for i in 1:N
        B_mat[i,   i  ] = real(hT[i+N, i+N])
        B_mat[i+N, i+N] = real(hT[i,   i  ])
    end
    mul!(tmp1, UT, B_mat)
    mul!(tmp2, tmp1, UT')
    _symmetrize_hermitian!(tmp2)
    Λ_init = copy(tmp2)
    E = real(dot(Λ_init, Γ_cur))

    # ── Adjoint gradient (backward pass) ──────────────────────────────────────
    Λ_cur  = copy(Λ_init)
    ∇E     = zeros(Float64, P)
    k_flat = P

    for s in D:-1:1
        stage = circuit[s]
        Us    = U_per_stage[s]

        if stage isa Tuple{<:AbstractMatrix, <:Real}
            # ── Separable stage: exact via [G_k, Γ_s] ─────────────────────
            Gk = gen_list[k_flat]
            Γs = Γ_per_stage[s]
            mul!(tmp1, Gk, Γs)
            mul!(tmp2, Γs, Gk)
            tmp1 .-= tmp2                      # [G_k, Γ_s]
            mul!(tmp2, Λ_cur, tmp1)
            ∇E[k_flat] = -2 * imag(tr(tmp2))
            k_flat -= 1
        else
            # ── Combined stage: exact Fréchet gradient ─────────────────────
            Gs, _, (D_H, VH) = stage
            d     = real.(diag(D_H))
            E_vec = exp.(2im .* d)
            _fréchet_phi!(Φbuf, d, E_vec)

            # Transform Λ and Γ_{s-1} to H-eigenbasis (once per stage)
            mul!(tmp1, VH', Λ_cur);              mul!(W,    tmp1, VH)
            mul!(tmp1, VH', Γ_prev_stage[s]);    mul!(G̃buf, tmp1, VH)

            for _ in length(Gs):-1:1
                Gk = gen_list[k_flat]
                mul!(tmp1, VH', Gk);  mul!(tmp2, tmp1, VH)   # tmp2 = G̃_k
                tmp2 .*= Φbuf                                 # tmp2 = Φ ⊙ G̃_k
                mul!(tmp1, W, tmp2)
                mul!(tmp2, tmp1, G̃buf)
                ∇E[k_flat] = -4 * imag(dot(E_vec, diag(tmp2)))
                k_flat -= 1
            end
        end

        # Pull co-state: Λ ← U_s'·Λ·U_s
        mul!(tmp1, Us', Λ_cur)
        mul!(tmp2, tmp1, Us)
        _symmetrize_hermitian!(tmp2)
        copyto!(Λ_cur, tmp2)
    end

    return_state ? (return E, ∇E, Γ_cur) : (return E, ∇E)
end


# ── Concrete wrappers ──────────────────────────────────────────────────────────

"""
    vgm_NN_Rxx_Rz(x, p; kwargs...)

Energy value, adjoint gradient, and tridiagonal FS metric for `f_loss_NN_Rxx_Rz`.
Circuit: 2D separable stages alternating `exp(2i·J·H_xx)`, `exp(2i·h·H_z)`.
Returns `(E, ∇x, g::BandedMatrix{Float64})`.
"""
function vgm_NN_Rxx_Rz(x::AbstractVector, p; kwargs...)
    ψ0, N, D, (hT, UT), return_state, _ = p
    G_xx = Matrix{ComplexF64}(TFI_Hamiltonian(N, 1.0, 0.0; PBC=0.0))
    G_z  = Matrix{ComplexF64}(TFI_Hamiltonian(N, 0.0, 1.0; PBC=0.0))
    circuit = [(isodd(s) ? G_xx : G_z, Float64(x[s])) for s in 1:2D]
    return value_gradient_and_metric(ψ0, circuit, hT, UT; return_state=return_state, kwargs...)
end


"""
    vgm_NN_TFI_only(x, p; kwargs...)

Energy value, adjoint gradient, and tridiagonal FS metric for `f_loss_NN_TFI_only`.
Circuit: D combined stages `exp(2i·(tJ·H_xx + h·H_z))`, 2 params each.
Metric uses the linearisation approximation `∂H/∂tJ = H_xx`, `∂H/∂h = H_z`.
Returns `(E, ∇x, g::BandedMatrix{Float64})`.
"""
function vgm_NN_TFI_only(x::AbstractVector, p; kwargs...)
    ψ0, N, D, (hT, UT), return_state, _ = p
    G_xx = Matrix{ComplexF64}(TFI_Hamiltonian(N, 1.0, 0.0; PBC=0.0))
    G_z  = Matrix{ComplexF64}(TFI_Hamiltonian(N, 0.0, 1.0; PBC=0.0))
    circuit = map(1:D) do d
        tJ = Float64(x[2d-1]); h = Float64(x[2d])
        ([G_xx, G_z], [tJ, h], Fu.Diag_h(tJ .* G_xx .+ h .* G_z))
    end
    return value_gradient_and_metric(ψ0, circuit, hT, UT; return_state=return_state, regularise=1e-6, kwargs...)
end


"""
    vgm_NN_TFI_brickwall(x, p; kwargs...)

Energy value, adjoint gradient, and tridiagonal FS metric for `f_loss_NN_TFI_brickwall`.
Circuit: 2D combined stages per layer (odd-bond then even-bond brickwall),
2 params each. Returns `(E, ∇x, g::BandedMatrix{Float64})`.
"""
function vgm_NN_TFI_brickwall(x::AbstractVector, p; kwargs...)
    ψ0, N, D, (hT, UT), return_state, _ = p
    G_z    = Matrix{ComplexF64}(TFI_Hamiltonian(N, 0.0, 1.0; PBC=0.0))
    odd_vec  = repeat([1.0, 0.0], N÷2)[1:N-1]
    even_vec = repeat([0.0, 1.0], N÷2)[1:N-1]
    G_odd  = Matrix{ComplexF64}(TFI_Hamiltonian(N, odd_vec,  zeros(N); PBC=0.0))
    G_even = Matrix{ComplexF64}(TFI_Hamiltonian(N, even_vec, zeros(N); PBC=0.0))
    circuit = mapreduce(vcat, 1:D) do d
        tJ1 = Float64(x[4d-3]); h1 = Float64(x[4d-2])
        tJ2 = Float64(x[4d-1]); h2 = Float64(x[4d  ])
        [
            ([G_odd,  G_z], [tJ1, h1], _safe_diag_h(tJ1 .* G_odd  .+ h1 .* G_z)),
            ([G_even, G_z], [tJ2, h2], _safe_diag_h(tJ2 .* G_even .+ h2 .* G_z))
        ]
    end
    return value_gradient_and_metric(ψ0, circuit, hT, UT; return_state=return_state, kwargs...)
end


"""
    vgm_NN_Kitaev_only(x, p; kwargs...)

Energy value, adjoint gradient, and tridiagonal FS metric for `f_loss_NN_Kitaev_only`.
Circuit: D combined stages `exp(2i·(tJ·H_hop + Δ·H_pair + μ·H_chem))`, 3 params each.
Returns `(E, ∇x, g::BandedMatrix{Float64})`.
"""
function vgm_NN_Kitaev_only(x::AbstractVector, p; kwargs...)
    ψ0, N, D, (hT, UT), return_state, _ = p
    G_hop  = Matrix{ComplexF64}(Kitaev_Hamiltonian(N, 1.0, 0.0, 0.0; PBC=0.0))
    G_pair = Matrix{ComplexF64}(Kitaev_Hamiltonian(N, 0.0, 1.0, 0.0; PBC=0.0))
    G_chem = Matrix{ComplexF64}(Kitaev_Hamiltonian(N, 0.0, 0.0, 1.0; PBC=0.0))
    circuit = map(1:D) do d
        tJ = Float64(x[3d-2]); Δ = Float64(x[3d-1]); μ = Float64(x[3d])
        ([G_hop, G_pair, G_chem], [tJ, Δ, μ],
         Fu.Diag_h(tJ .* G_hop .+ Δ .* G_pair .+ μ .* G_chem))
    end
    return value_gradient_and_metric(ψ0, circuit, hT, UT; return_state=return_state,kwargs...)
end


"""
    vgm_NN_Kitaev_HVA(x, p; kwargs...)

Energy value, adjoint gradient, and tridiagonal FS metric for `f_loss_NN_Kitaev_HVA`.
Circuit: 3D separable stages cycling `exp(2i·tJ·H_hop)`, `exp(2i·Δ·H_pair)`,
`exp(2i·μ·H_chem)`. Returns `(E, ∇x, g::BandedMatrix{Float64})`.
"""
function vgm_NN_Kitaev_HVA(x::AbstractVector, p; kwargs...)
    ψ0, N, D, (hT, UT), return_state, _ = p
    G_hop  = Matrix{ComplexF64}(Kitaev_Hamiltonian(N, 1.0, 0.0, 0.0; PBC=0.0))
    G_pair = Matrix{ComplexF64}(Kitaev_Hamiltonian(N, 0.0, 1.0, 0.0; PBC=0.0))
    G_chem = Matrix{ComplexF64}(Kitaev_Hamiltonian(N, 0.0, 0.0, 1.0; PBC=0.0))
    Gs = (G_hop, G_pair, G_chem)
    circuit = [(Gs[mod1(s, 3)], Float64(x[s])) for s in 1:3D]
    return value_gradient_and_metric(ψ0, circuit, hT, UT; return_state=return_state, kwargs...)
end


"""
    vgm_EXP_HVA(x, p; kwargs...)

Energy value, adjoint gradient, and FS metric for an HVA-style circuit targeting
the TFI with exponentially decaying XX interactions and a separated transverse field.
Each layer contributes two stages:
  1. Combined stage `exp(2i·H_XX(λ,tJ))` with H_XX = tJ·G_XX(λ), 2 params (λ, tJ).
     λ enters non-linearly; gradient uses Fréchet/Duhamel with
     G_λ = ∂H_XX/∂λ and G_tJ = ∂H_XX/∂tJ.
  2. Separable stage `exp(2i·hoverJ·G_z)`, 1 param (hoverJ).
Parameter layout per layer d: x[3d-2]=λ, x[3d-1]=tJ, x[3d]=hoverJ. Total P=3D.
Returns `(E, ∇x, g)`.
"""
function vgm_EXP_HVA(x::AbstractVector, p; kwargs...)
    ψ0, N, D, (hT, UT), return_state, _ = p
    G_z = Matrix{ComplexF64}(LongRangeTFI_Hamiltonian(N, zeros(N*(N-1)÷2), ones(N)))
    circuit = mapreduce(vcat, 1:D) do d
        lambda  = Float64(x[3d-2])
        tJ      = Float64(x[3d-1])
        hoverJ  = Float64(x[3d])
        H_XX    = LongRangeTFI_exp_Hamiltonian(N, tJ, 0.0, lambda)
        G_lambda = Matrix{ComplexF64}(LongRangeTFI_Hamiltonian(N,
            build_dexp_dlambda(N, tJ, lambda), zeros(N)))
        G_tJ     = Matrix{ComplexF64}(LongRangeTFI_Hamiltonian(N,
            build_exp_couplings(N, 1.0, lambda), zeros(N)))
        [
            ([G_lambda, G_tJ], [lambda, tJ], Fu.Diag_h(Matrix(H_XX))),
            (G_z, hoverJ)
        ]
    end
    return value_gradient_and_metric(ψ0, circuit, hT, UT; return_state=return_state, kwargs...)
end


"""
    vgm_POW_HVA(x, p; kwargs...)

Energy value, adjoint gradient, and FS metric for an HVA-style circuit targeting
the TFI with power-law decaying XX interactions and a separated transverse field.
Each layer contributes two stages:
  1. Combined stage `exp(2i·H_XX(α,tJ))` with H_XX = tJ·G_XX(α), 2 params (α, tJ).
     α enters non-linearly; gradient uses Fréchet/Duhamel with
     G_α = ∂H_XX/∂α and G_tJ = ∂H_XX/∂tJ.
  2. Separable stage `exp(2i·hoverJ·G_z)`, 1 param (hoverJ).
Parameter layout per layer d: x[3d-2]=α, x[3d-1]=tJ, x[3d]=hoverJ. Total P=3D.
Returns `(E, ∇x, g)`.
"""
function vgm_POW_HVA(x::AbstractVector, p; kwargs...)
    ψ0, N, D, (hT, UT), return_state, _ = p
    G_z = Matrix{ComplexF64}(LongRangeTFI_Hamiltonian(N, zeros(N*(N-1)÷2), ones(N)))
    circuit = mapreduce(vcat, 1:D) do d
        alpha   = Float64(x[3d-2])
        tJ      = Float64(x[3d-1])
        hoverJ  = Float64(x[3d])
        H_XX    = LongRangeTFI_pow_Hamiltonian(N, tJ, 0.0, alpha)
        G_alpha  = Matrix{ComplexF64}(LongRangeTFI_Hamiltonian(N,
            build_dpow_dalpha(N, tJ, alpha), zeros(N)))
        G_tJ     = Matrix{ComplexF64}(LongRangeTFI_Hamiltonian(N,
            build_pow_couplings(N, 1.0, alpha), zeros(N)))
        [
            ([G_alpha, G_tJ], [alpha, tJ], Fu.Diag_h(Matrix(H_XX))),
            (G_z, hoverJ)
        ]
    end
    return value_gradient_and_metric(ψ0, circuit, hT, UT; return_state=return_state, kwargs...)
end


"""
    vgm_LR_EXP_TFI_only(x, p; kwargs...)

Energy value, adjoint Frechet gradient, and FS metric for `f_loss_EXP_TFI_only`.
Circuit: D combined stages `exp(2i·H_EXP(tJ, hoverJ, lambda))`, 3 params each:
  x[3d-2] = lambda   (exponential decay rate)
  x[3d-1] = tJ       (overall coupling prefactor)
  x[3d]   = hoverJ   (transverse field)
Generators:
  G_lambda  = dH/dlambda = LongRangeTFI(N, dJ_dlambda(lambda), 0)
  G_tJ      = dH/dtJ     = LongRangeTFI(N, exp_couplings(1.0, lambda), 0)
  G_hoverJ  = dH/dhoverJ = LongRangeTFI(N, 0, 1)   (pure Z field)
Returns `(E, grad, g, state_final)` or `(E, grad, g)`.
"""
function vgm_LR_EXP_TFI_only(x::AbstractVector, p; kwargs...)
    ψ0, N, D, (hT, UT), return_state, _ = p
    G_z = Matrix{ComplexF64}(LongRangeTFI_Hamiltonian(N, zeros(N*(N-1)÷2), ones(N)))
    circuit = map(1:D) do d
        lambda  = Float64(x[3d-2])
        tJ      = Float64(x[3d-1])
        hoverJ  = Float64(x[3d])
        H_stage = LongRangeTFI_exp_Hamiltonian(N, tJ, hoverJ, lambda)
        G_lambda = Matrix{ComplexF64}(LongRangeTFI_Hamiltonian(N,
            build_dexp_dlambda(N, tJ, lambda), zeros(N)))
        G_tJ     = Matrix{ComplexF64}(LongRangeTFI_Hamiltonian(N,
            build_exp_couplings(N, 1.0, lambda), zeros(N)))
        ([G_lambda, G_tJ, G_z], [lambda, tJ, hoverJ], Fu.Diag_h(Matrix(H_stage)))
    end
    return value_gradient_and_metric(ψ0, circuit, hT, UT; return_state=return_state, kwargs...)
end


"""
    vgm_LR_POW_TFI_only(x, p; kwargs...)

Energy value, adjoint Frechet gradient, and FS metric for `f_loss_POW_TFI_only`.
Circuit: D combined stages `exp(2i·H_POW(tJ, hoverJ, alpha))`, 3 params each:
  x[3d-2] = alpha     (power-law exponent)
  x[3d-1] = tJ        (overall coupling prefactor)
  x[3d]   = hoverJ    (transverse field)
Generators:
  G_alpha  = dH/dalpha  = LongRangeTFI(N, -tJ*log(d)/d^alpha per pair, 0)
  G_tJ     = dH/dtJ     = LongRangeTFI(N, pow_couplings(1.0, alpha), 0)
  G_hoverJ = dH/dhoverJ = LongRangeTFI(N, 0, 1)   (pure Z field)
Returns `(E, grad, g, state_final)` or `(E, grad, g)`.
"""
function vgm_LR_POW_TFI_only(x::AbstractVector, p; kwargs...)
    ψ0, N, D, (hT, UT), return_state, _ = p
    G_z = Matrix{ComplexF64}(LongRangeTFI_Hamiltonian(N, zeros(N*(N-1)÷2), ones(N)))
    circuit = map(1:D) do d
        alpha   = Float64(x[3d-2])
        tJ      = Float64(x[3d-1])
        hoverJ  = Float64(x[3d])
        H_stage = LongRangeTFI_pow_Hamiltonian(N, tJ, hoverJ, alpha)
        G_alpha = Matrix{ComplexF64}(LongRangeTFI_Hamiltonian(N,
            build_dpow_dalpha(N, tJ, alpha), zeros(N)))
        G_tJ    = Matrix{ComplexF64}(LongRangeTFI_Hamiltonian(N,
            build_pow_couplings(N, 1.0, alpha), zeros(N)))
        ([G_alpha, G_tJ, G_z], [alpha, tJ, hoverJ], Fu.Diag_h(Matrix(H_stage)))
    end
    return value_gradient_and_metric(ψ0, circuit, hT, UT; return_state=return_state, kwargs...)
end


# 
#  Metric-free wrappers (vg_*)                                                
#  
#  Each `vg_*` is the same circuit as its `vgm_*` sibling, but calls   
#  `value_and_gradient` instead of `value_gradient_and_metric`.  Returns
#  `(E, ∇E)` or `(E, ∇E, Γ_final)` if `return_state` is true in `p`.
#  Drops all metric-only kwargs (`regularise`, `full_metric`, `natural_grad`).
#  Cost: O(D·N³) instead of O(P²·N³).                          


"""
    vg_NN_Rxx_Rz(x, p) — see `vgm_NN_Rxx_Rz`. Returns `(E, ∇E)` only.
"""
function vg_NN_Rxx_Rz(x::AbstractVector, p; return_state::Bool=false)
    ψ0, N, D, (hT, UT), _, _ = p
    G_xx = Matrix{ComplexF64}(TFI_Hamiltonian(N, 1.0, 0.0; PBC=0.0))
    G_z  = Matrix{ComplexF64}(TFI_Hamiltonian(N, 0.0, 1.0; PBC=0.0))
    circuit = [(isodd(s) ? G_xx : G_z, Float64(x[s])) for s in 1:2D]
    return value_and_gradient(ψ0, circuit, hT, UT; return_state=return_state)
end


"""
    vg_NN_TFI_only(x, p) — see `vgm_NN_TFI_only`. Returns `(E, ∇E)` only.
"""
function vg_NN_TFI_only(x::AbstractVector, p; return_state::Bool=false)
    ψ0, N, D, (hT, UT), _, _ = p
    G_xx = Matrix{ComplexF64}(TFI_Hamiltonian(N, 1.0, 0.0; PBC=0.0))
    G_z  = Matrix{ComplexF64}(TFI_Hamiltonian(N, 0.0, 1.0; PBC=0.0))
    circuit = map(1:D) do d
        tJ = Float64(x[2d-1]); h = Float64(x[2d])
        ([G_xx, G_z], [tJ, h], Fu.Diag_h(tJ .* G_xx .+ h .* G_z))
    end
    return value_and_gradient(ψ0, circuit, hT, UT; return_state=return_state)
end


"""
    vg_NN_Kitaev_only(x, p) — see `vgm_NN_Kitaev_only`. Returns `(E, ∇E)` only.
"""
function vg_NN_Kitaev_only(x::AbstractVector, p; return_state::Bool=false)
    ψ0, N, D, (hT, UT), _, _ = p
    G_hop  = Matrix{ComplexF64}(Kitaev_Hamiltonian(N, 1.0, 0.0, 0.0; PBC=0.0))
    G_pair = Matrix{ComplexF64}(Kitaev_Hamiltonian(N, 0.0, 1.0, 0.0; PBC=0.0))
    G_chem = Matrix{ComplexF64}(Kitaev_Hamiltonian(N, 0.0, 0.0, 1.0; PBC=0.0))
    circuit = map(1:D) do d
        tJ = Float64(x[3d-2]); Δ = Float64(x[3d-1]); μ = Float64(x[3d])
        ([G_hop, G_pair, G_chem], [tJ, Δ, μ],
         Fu.Diag_h(tJ .* G_hop .+ Δ .* G_pair .+ μ .* G_chem))
    end
    return value_and_gradient(ψ0, circuit, hT, UT; return_state=return_state)
end


"""
    vg_NN_Kitaev_HVA(x, p) — see `vgm_NN_Kitaev_HVA`. Returns `(E, ∇E)` only.
"""
function vg_NN_Kitaev_HVA(x::AbstractVector, p; return_state::Bool=false)
    ψ0, N, D, (hT, UT), _, _ = p
    G_hop  = Matrix{ComplexF64}(Kitaev_Hamiltonian(N, 1.0, 0.0, 0.0; PBC=0.0))
    G_pair = Matrix{ComplexF64}(Kitaev_Hamiltonian(N, 0.0, 1.0, 0.0; PBC=0.0))
    G_chem = Matrix{ComplexF64}(Kitaev_Hamiltonian(N, 0.0, 0.0, 1.0; PBC=0.0))
    Gs = (G_hop, G_pair, G_chem)
    circuit = [(Gs[mod1(s, 3)], Float64(x[s])) for s in 1:3D]
    return value_and_gradient(ψ0, circuit, hT, UT; return_state=return_state)
end


"""
    vg_EXP_HVA(x, p) — see `vgm_EXP_HVA`. Returns `(E, ∇E)` only.
"""
function vg_EXP_HVA(x::AbstractVector, p; return_state::Bool=false)
    ψ0, N, D, (hT, UT), _, _ = p
    G_z = Matrix{ComplexF64}(LongRangeTFI_Hamiltonian(N, zeros(N*(N-1)÷2), ones(N)))
    circuit = mapreduce(vcat, 1:D) do d
        lambda  = Float64(x[3d-2])
        tJ      = Float64(x[3d-1])
        hoverJ  = Float64(x[3d])
        H_XX    = LongRangeTFI_exp_Hamiltonian(N, tJ, 0.0, lambda)
        G_lambda = Matrix{ComplexF64}(LongRangeTFI_Hamiltonian(N,
            build_dexp_dlambda(N, tJ, lambda), zeros(N)))
        G_tJ     = Matrix{ComplexF64}(LongRangeTFI_Hamiltonian(N,
            build_exp_couplings(N, 1.0, lambda), zeros(N)))
        [
            ([G_lambda, G_tJ], [lambda, tJ], Fu.Diag_h(Matrix(H_XX))),
            (G_z, hoverJ)
        ]
    end
    return value_and_gradient(ψ0, circuit, hT, UT; return_state=return_state)
end


"""
    vg_POW_HVA(x, p) — see `vgm_POW_HVA`. Returns `(E, ∇E)` only.
"""
function vg_POW_HVA(x::AbstractVector, p; return_state::Bool=false)
    ψ0, N, D, (hT, UT), _, _ = p
    G_z = Matrix{ComplexF64}(LongRangeTFI_Hamiltonian(N, zeros(N*(N-1)÷2), ones(N)))
    circuit = mapreduce(vcat, 1:D) do d
        alpha   = Float64(x[3d-2])
        tJ      = Float64(x[3d-1])
        hoverJ  = Float64(x[3d])
        H_XX    = LongRangeTFI_pow_Hamiltonian(N, tJ, 0.0, alpha)
        G_alpha  = Matrix{ComplexF64}(LongRangeTFI_Hamiltonian(N,
            build_dpow_dalpha(N, tJ, alpha), zeros(N)))
        G_tJ     = Matrix{ComplexF64}(LongRangeTFI_Hamiltonian(N,
            build_pow_couplings(N, 1.0, alpha), zeros(N)))
        [
            ([G_alpha, G_tJ], [alpha, tJ], Fu.Diag_h(Matrix(H_XX))),
            (G_z, hoverJ)
        ]
    end
    return value_and_gradient(ψ0, circuit, hT, UT; return_state=return_state)
end


"""
    vg_LR_EXP_TFI_only(x, p) — see `vgm_LR_EXP_TFI_only`. Returns `(E, ∇E)` only.
"""
function vg_LR_EXP_TFI_only(x::AbstractVector, p; return_state::Bool=false)
    ψ0, N, D, (hT, UT), _, _ = p
    G_z = Matrix{ComplexF64}(LongRangeTFI_Hamiltonian(N, zeros(N*(N-1)÷2), ones(N)))
    circuit = map(1:D) do d
        lambda  = Float64(x[3d-2])
        tJ      = Float64(x[3d-1])
        hoverJ  = Float64(x[3d])
        H_stage = LongRangeTFI_exp_Hamiltonian(N, tJ, hoverJ, lambda)
        G_lambda = Matrix{ComplexF64}(LongRangeTFI_Hamiltonian(N,
            build_dexp_dlambda(N, tJ, lambda), zeros(N)))
        G_tJ     = Matrix{ComplexF64}(LongRangeTFI_Hamiltonian(N,
            build_exp_couplings(N, 1.0, lambda), zeros(N)))
        ([G_lambda, G_tJ, G_z], [lambda, tJ, hoverJ], Fu.Diag_h(Matrix(H_stage)))
    end
    return value_and_gradient(ψ0, circuit, hT, UT; return_state=return_state)
end


"""
    vg_LR_POW_TFI_only(x, p) — see `vgm_LR_POW_TFI_only`. Returns `(E, ∇E)` only.
"""
function vg_LR_POW_TFI_only(x::AbstractVector, p; return_state::Bool=false)
    ψ0, N, D, (hT, UT), _, _ = p
    G_z = Matrix{ComplexF64}(LongRangeTFI_Hamiltonian(N, zeros(N*(N-1)÷2), ones(N)))
    circuit = map(1:D) do d
        alpha   = Float64(x[3d-2])
        tJ      = Float64(x[3d-1])
        hoverJ  = Float64(x[3d])
        H_stage = LongRangeTFI_pow_Hamiltonian(N, tJ, hoverJ, alpha)
        G_alpha = Matrix{ComplexF64}(LongRangeTFI_Hamiltonian(N,
            build_dpow_dalpha(N, tJ, alpha), zeros(N)))
        G_tJ    = Matrix{ComplexF64}(LongRangeTFI_Hamiltonian(N,
            build_pow_couplings(N, 1.0, alpha), zeros(N)))
        ([G_alpha, G_tJ, G_z], [alpha, tJ, hoverJ], Fu.Diag_h(Matrix(H_stage)))
    end
    return value_and_gradient(ψ0, circuit, hT, UT; return_state=return_state)
end




function sigma_x_correlator_N2(Γ::AbstractMatrix{<:Number}, N::Int)
    @assert size(Γ) == (2N, 2N) "Γ must be 2N×2N, got $(size(Γ))"

    m = N ÷ 2  # reference site (1-indexed)

    # Kernel f(k, l) = ⟨B_k A_l⟩, k and l are 1-indexed site labels
    # B_k = c_k - c†_k,  A_l = c_l + c†_l
    # f(k,l) = ⟨c_k c_l⟩ - ⟨c†_k c_l⟩ + ⟨c_k c†_l⟩ - ⟨c†_k c†_l⟩
    #        = Γ[k+N, l]  - Γ[k, l]    + Γ[k+N, l+N] - Γ[k, l+N]
    f(k, l) = Γ[k+N, l] - Γ[k, l] + Γ[k+N, l+N] - Γ[k, l+N]

    xs = (-(N÷2 - 1)):(N÷2)

    Cxx = map(xs) do x
        n = m + x  # target site

        x == 0 && return 1.0 + 0.0im  # ⟨(σˣ)²⟩ = 1, det of 0×0 matrix

        # Build the |x| × |x| Toeplitz-like matrix
        # For x > 0: string from m to n=m+x
        #   T[j,l] = f(m+j-1, m+l),  j,l = 1,…,x
        #   (rows: B operators at sites m, m+1, …, n-1)
        #   (cols: A operators at sites m+1, m+2, …, n)
        # For x < 0: string from n to m, same formula with m↔n
        lo, hi = x > 0 ? (m, n) : (n, m)
        r = hi - lo  # string length = |x|

        T = [f(lo + j - 1, lo + l) for j in 1:r, l in 1:r]
        det(T)
    end

    return Cxx
end



function single_eval_Ising(
    state_final::AbstractMatrix{<:Number},
    # H_target::AbstractMatrix{<:Number},
    U_target::AbstractMatrix{<:Number},
    # HD_target::AbstractMatrix{<:Real},
    Γ_target::AbstractMatrix{<:Number},
    final_energy::Float64,
    final_params::Vector{Float64},
    N::Int,
    D::Int,
  )
    N_A = N ÷ 2

    #### Eigenmodes and entanglement spectrum 
    n_k = real(diag(U_target' * state_final * U_target))
    # variance_H = hamiltonian_variance(state_final, HD_target, U_target)
    final_entropy = Fu.VN_entropy(Fu.Reduce_gamma(state_final, N_A, 1))
    ES_vals_even, ES_vals_odd = ES_spec(state_final)
    ES_ratios_even, ES_ratios_odd = ES_spec_ratios(ES_vals_even, ES_vals_odd)

    ###### Entanglement profiles S(ℓ) for ℓ = 1, ..., N/2 
    S_profile_circuit = [Fu.VN_entropy(Fu.Reduce_gamma(state_final, ℓ, 1)) for ℓ in 1:N_A]
    S_profile_target  = [Fu.VN_entropy(Fu.Reduce_gamma(Γ_target, ℓ, 1)) for ℓ in 1:N_A]

    ###### Entanglement penetration depth from profile fit 
    efit = fit_boundary_cutoff_robust(S_profile_circuit, N)
    ξ_ent = efit.ξ_ent
    c_eff = efit.c_eff
    plateau_start = efit.plateau_start
    

    C_xx = sigma_x_correlator_N2(state_final,N)

    return (;
      N, D, final_params,
      final_energy,
      n_k,
      final_entropy, S_profile_circuit, S_profile_target,
      ES_ratios_even, ES_ratios_odd, ES_vals_even, ES_vals_odd,
      ξ_ent, c_eff, plateau_start,
      C_xx
    )
end

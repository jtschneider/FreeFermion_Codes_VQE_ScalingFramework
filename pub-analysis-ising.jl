# Standalone Ising publication analysis extracted from notebooks/pub-analysis.jl.

ENV["MKL_DEBUG_CPU_TYPE"] = "5"
ENV["BLIS_NUM_THREADS"] = "4"

# Repository root. Override with FREE_FERMIONS_BASE_DIR when the notebook is
# executed from a copied or relocated directory.
const BASE_DIR = normpath(get(
    ENV,
    "FREE_FERMIONS_BASE_DIR",
    joinpath(@__DIR__, "."),
))

# import Pkg; Pkg.activate(BASE_DIR)

begin
    using LinearAlgebra, Random, Printf, Statistics
    using Optim, NLSolversBase
    import F_utilities as Fu
    using HDF5, JLD2
    using BLISBLAS
    using LaTeXStrings
    using LsqFit
    using Makie, CairoMakie
    using Markdown
end


##%
CairoMakie.activate!(type = "png", px_per_unit = 4)
include(joinpath(BASE_DIR, "free_fermions_AD.jl"))

LinearAlgebra.BLAS.set_num_threads(Threads.nthreads())
BLISBLAS.set_num_threads(Threads.nthreads())


##%
# Plot theme and global analysis grids
# Source Pluto cell: dd3d7284-f38f-4373-98b5-f4a69c3aa9a2
read_for_print_theme = Theme(
    # fonts          = (; regular = "TeX Gyre Termes Math", bold = "TeX Gyre Termes Bold"),
	# fonts          = (; regular = "NewComputerModernMath", bold = "LMRoman12"),
	fonts = (; regular = "Latin Modern Math", bold = "LMRoman10 Bold"),
    fontsize       = 10,
    size           = (320, 200),
    figure_padding = 1,
    Axis = (
        xlabelpadding      = 2,
        xminorgridvisible  = true,
        xminorticksvisible = true,
        xminorticks        = IntervalsBetween(9),
        ylabelpadding      = 2,
        yminorgridvisible  = true,
        yminorticksvisible = true,
        yminorticks        = IntervalsBetween(9),
    ),
); set_theme!(read_for_print_theme)


# Source Pluto cell: dbdae251-3268-48a6-9d30-067ce9bf79bb
E_OBC(N) = (1.0 - csc(π/(4N + 2)))


# Source Pluto cell: 6a0a9985-5e74-4eac-846e-398c9462036b
dispersion_relation(k,N) = -2cos(k*π/(2N+1))


# Source Pluto cell: 633893c5-416f-4c48-9a7b-0a63e42da282
D_scan = [1:12..., 14:2:22...]


# Source Pluto cell: 420517b2-6eef-47c4-b607-e9c0b630fabb
N_scan = [8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 128, 256, 512]
# N_scan = [8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 128, 156, 184, 200, 240, 256, 280, 300, 320, 360, 380, 400, 512, 620, 660, 700, 740]


# Source Pluto cell: b90c2135-c137-4bae-b30e-076fc0a6be72
loop_keys = [:TFI_only, :Rxx_Rz, :kitaev_only, :kitaev_HVA, :TFI_EXP, :EXP_HVA, :TFI_POW, :POW_HVA]


##%
# Ansatz definitions
# Source Pluto cell: d71d61d7-293b-4b8f-a836-d1168500727c
const VARIANTS = (
    Rxx_Rz=(
        n_params=2,
        params_labels=["𝑡𝐽", "𝜃"],
        label="TFIM HVA",
        fun=vgm_NN_Rxx_Rz,
    ),
    TFI_only=(
        n_params=2,
        params_labels=["𝑡𝐽", "ℎ/𝐽"],
        label="TFIM combined",
        fun=vgm_NN_TFI_only,
    ),
    TFI_EXP=(
        n_params=3,
        params_labels=["𝜆", "𝑡𝐽", "ℎ/𝐽"],
        label="EXP combined",
        fun=vgm_LR_EXP_TFI_only,
    ),
    EXP_HVA=(
        n_params=3,
        params_labels=["𝜆", "𝑡𝐽", "ℎ/𝐽"],
        label="EXP HVA",
        fun=vgm_EXP_HVA,
    ),
    POW_HVA=(
        n_params=3,
        params_labels=["𝛼", "𝑡𝐽", "ℎ/𝐽"],
        label="POW HVA",
        fun=vgm_POW_HVA,
    ),
    TFI_POW=(
        n_params=3,
        params_labels=["𝛼", "𝑡𝐽", "ℎ/𝐽"],
        label="POW combined",
        fun=vgm_LR_POW_TFI_only,
    ),
    kitaev_only=(
        n_params=3,
        params_labels=["𝑡𝐽", "𝑡Δ", "𝑡𝜇"],
        label="Kitaev combined",
        fun=vgm_NN_Kitaev_only,
    ),
    kitaev_HVA=(
        n_params=3,
        params_labels=["𝑡𝐽", "𝑡Δ", "𝑡𝜇"],
        label="Kitaev HVA",
        fun=vgm_NN_Kitaev_HVA,
    )
)


# Source Pluto cell: 1aa2b7af-14ac-4e42-84cc-1a1bdbba73de
N_mode_plot = 64


##%
# Results container and HDF5 loader
# Source Pluto cell: 04adb39f-57d2-48a4-9dfb-a870ab518883
"""
Stores everything a single-ansatz analysis this script needs to make the
publication plots. Maps 1-to-1 onto the layout
`<ansatz_key>/N_<N>/D_<D>/<observable>` inside the unified HDF5 file.
"""
struct CircuitResults
    ### Axes 
    Ns ::Vector{Int}
    Ds ::Vector{Int}

    ### Per-(N,D) scalars: Ns × Ds matrices 
    energies       ::Matrix{Float64}
    final_entropy  ::Matrix{Float64}     # half-chain S
    ξ_ent          ::Matrix{Float64}
    c_eff          ::Matrix{Float64}
    plateau_start  ::Matrix{Int}

    #### Per-(N,D) variable-length vectors: Dicts 
    params              ::Dict{Tuple{Int,Int}, Vector{Float64}}
    # eigenmodes occupations:
    n_k                 ::Dict{Tuple{Int,Int}, Vector{Float64}}  
    # S_profiles_circuit  ::Dict{Tuple{Int,Int}, Vector{Float64}}
    # S_profiles_target   ::Dict{Tuple{Int,Int}, Vector{Float64}}
    # ES_vals_even        ::Dict{Tuple{Int,Int}, Vector{Float64}}
    # ES_vals_odd         ::Dict{Tuple{Int,Int}, Vector{Float64}}
    # ES_ratios_even      ::Dict{Tuple{Int,Int}, Vector{Float64}}
    # ES_ratios_odd       ::Dict{Tuple{Int,Int}, Vector{Float64}}
    correlator  ::Dict{Tuple{Int,Int}, Vector{Float64}}
    correlator_2  ::Dict{Tuple{Int,Int}, Vector{Float64}}

    ### Derived once at load time (cached for plotting speed) 
    δenergy_density ::Matrix{Float64}    # energies./Ns .+ 4/π
    δE0             ::Matrix{Float64}    # |energies - E_exact|
    pert_data       ::Matrix{Float64}    # Ns .* δE0 ./ (2π)
    density_pert    ::Matrix{Float64}    # abs.((δenergy_density .- (1.0 - 2.0/π) ./N_scan[iNs_FSS]) .* (N_scan[iNs_FSS] .^2))

    ### Regime filters (filled in by classify_regimes_fit) 
    # FSS_filter ::Matrix{Bool}
    FSS_filter_v2 ::Matrix{Bool}
    # FDS_filter ::Matrix{Bool}
    # CO_filter  ::Matrix{Bool}
end



# Source Pluto cell: 1d7dfb25-ad34-4863-a1cb-13e2c97a5bc5
"""
    load_circuit_results(h5_file, key, Ns, Ds; E_exact_fn = E_OBC)

Pull every (N, D) entry under `<key>/N_<N>/D_<D>/…` out of an opened HDF5
file and assemble a `CircuitResults`. Missing groups are silently skipped
(the corresponding scalar matrix entries stay NaN; the Dicts simply do not
get a key). Derived fields use `E_exact_fn(N)` for the exact finite-size
ground state energy (default: Ising OBC).
"""
function load_circuit_results(h5_file::HDF5.File, key::Symbol,
                               Ns::AbstractVector{Int}, Ds::AbstractVector{Int};
                               E_exact_fn = E_OBC)
    nN, nD = length(Ns), length(Ds)

    energies       = fill(NaN, nN, nD)
    final_entropy  = fill(NaN, nN, nD)
    ξ_ent          = fill(NaN, nN, nD)
    c_eff          = fill(NaN, nN, nD)
    plateau_start  = fill(0,   nN, nD)

    params              = Dict{Tuple{Int,Int}, Vector{Float64}}()
    n_k                 = Dict{Tuple{Int,Int}, Vector{Float64}}()
    # S_profiles_circuit  = Dict{Tuple{Int,Int}, Vector{Float64}}()
    # S_profiles_target   = Dict{Tuple{Int,Int}, Vector{Float64}}()
    # ES_vals_even        = Dict{Tuple{Int,Int}, Vector{Float64}}()
    # ES_vals_odd         = Dict{Tuple{Int,Int}, Vector{Float64}}()
    # ES_ratios_even      = Dict{Tuple{Int,Int}, Vector{Float64}}()
    # ES_ratios_odd       = Dict{Tuple{Int,Int}, Vector{Float64}}()
    sigma_x_correlator  = Dict{Tuple{Int,Int}, Vector{Float64}}()
    empty_corr  = Dict{Tuple{Int,Int}, Vector{Float64}}()
    
    key_str = string(key)
    if !haskey(h5_file, key_str)
        @warn "no group for $(key) in HDF5 file — returning empty result"
    else
        for (iN, N) in enumerate(Ns), (iD, D) in enumerate(Ds)
            grp_path = string(key_str, "/N_", N, "/D_", D)
            haskey(h5_file, grp_path) || continue
            grp = h5_file[grp_path]

            energies[iN, iD]      = read(grp, "final_energy")
            final_entropy[iN, iD] = read(grp, "final_entropy")
            ξ_ent[iN, iD]         = read(grp, "ξ_ent")
            c_eff[iN, iD]         = read(grp, "c_eff")
            plateau_start[iN, iD] = Int(read(grp, "plateau_start"))

            params[(N, D)]             = read(grp, "final_params")
            if N == N_mode_plot
                n_k[(N, D)]                = read(grp, "n_k")
            end
            # S_profiles_circuit[(N, D)] = read(grp, "S_profile_circuit")
            # S_profiles_target[(N, D)]  = read(grp, "S_profile_target")
            # ES_vals_even[(N, D)]       = read(grp, "ES_vals_even")
            # ES_vals_odd[(N, D)]        = read(grp, "ES_vals_odd")
            # ES_ratios_even[(N, D)]     = read(grp, "ES_ratios_even")
            # ES_ratios_odd[(N, D)]      = read(grp, "ES_ratios_odd")
            sigma_x_correlator[(N, D)] = real.(read(grp, "C_xx"))[N÷2+1:end]
        end
    end

    # Derived fields (cached for plot performance).
    Ns_mat          = repeat(collect(Ns), 1, nD)
    E_exact         = map(E_exact_fn, Ns_mat)
    δenergy_density = energies ./ Ns_mat .+ 4.0/π
    δE0             = abs.(energies .- E_exact)
    pert_data       = Ns_mat .* δE0 ./ (2π)

    density_pert = abs.((δenergy_density .- (1.0 - 2.0/π) ./Ns_mat) .* (Ns_mat .^2))

    # Filters initialised by the corner anti-diagonal pattern; overwritten
    # later by `classify_regimes_fit`.
    FSS_filter, FDS_filter, CO_filter = falses(length(Ns), length(Ds)), falses(length(Ns), length(Ds)), falses(length(Ns), length(Ds))

    return CircuitResults(
        collect(Ns), collect(Ds),
        energies, final_entropy, ξ_ent, c_eff, plateau_start,
        params, n_k,
        # S_profiles_circuit, S_profiles_target,
        # ES_vals_even, ES_vals_odd,
        # ES_ratios_even, ES_ratios_odd,
        sigma_x_correlator, empty_corr,
        δenergy_density, δE0, pert_data, density_pert,
        FSS_filter, #FSS_filter, FDS_filter, CO_filter,
    )
end


##%
# Input/output paths and data loading
# Source Pluto cell: 6caa270c-9c43-4b5c-b54a-5ed12d0db37a
# const OUTPUT_FILE = joinpath("..", "optim_results_QNG_EucGradDesc.h5")
const OUTPUT_FILE = get(ENV, "ISING_RESULTS_FILE", let
    fallback = joinpath(BASE_DIR, "optim_results_Ising_QNG_EucGradDesc.h5")
    preferred = joinpath(BASE_DIR, "optim_results_QNG_EucGradDesc.h5")
    isfile(preferred) ? preferred : fallback
end)


# Source Pluto cell: 76e3b51f-b50a-4134-8444-df701a0e3581
const plotdir = joinpath(BASE_DIR, "Plots")


# Source Pluto cell: d9c76b32-4652-46b3-a2c0-7137833cc209
begin
	results = Dict{Symbol, CircuitResults}()

	h5open(OUTPUT_FILE, "r") do fh
        for key in loop_keys
            results[key] = load_circuit_results(fh, key, N_scan, D_scan)
            n_loaded = sum(isfinite, results[key].energies)
            println("loaded $(rpad(string(key), 14)) → $(n_loaded) (N,D) cells")
        end
    end

end


##%
# Finite-size-scaling classification
# Source Pluto cell: 5057f7c5-712f-4595-9e66-2ad09da3d535

"""
    classify_FSS!(res::CircuitResults;
                  fss_tol         = 0.08,
                  boundary_e0     = 1.0 - 2.0/π,
                  casimir_plateau = π/24,
                  two_sided_FSS   = false) -> CircuitResults

Tag the finite-size-scaling (FSS) regime of a single `CircuitResults` from the
**Casimir plateau of the bulk density perturbation**, writing the result into
`res.FSS_filter_v2` **in place** (hence the `!`). No `κ` is required — FSS
detection is purely a property of the energy density vs. the Casimir constant.

Bulk density perturbation
    density_pert(N,D) = | N²·(δe₀(N,D) − boundary_e0/N) |  =  | e_c + N·δE | ,
with the Casimir constant `e_c = −casimir_plateau`. In the FSS regime the VQE
reproduces the exact finite-size ground state (`N·δE → 0`), so
`density_pert → |e_c| = casimir_plateau` — a clean plateau.

**FSS detection (one-sided, default):** a point is FSS iff its `density_pert`
lies at or below the Casimir scale within a log-space margin,
    log(density_pert) − log(casimir_plateau) < log(1 + fss_tol),
keeping the plateau *and* the crossover dip (both ≤ plateau) as one connected
small-`L/D^κ` region; the `x²` FDS arm (≫ plateau) is the complement.
Set `two_sided_FSS = true` to require
`|log(density_pert/casimir_plateau)| < log(1+fss_tol)` (plateau only, dip out).

Override `casimir_plateau` for non-TFI normalizations (e.g. `π/12` for the
U(1)/XY models). Only `res.FSS_filter_v2` is modified; FDS/CO/κ are untouched.
"""
function classify_FSS!(res::CircuitResults;
                       fss_tol         = 0.08,
                       boundary_e0     = 1.0 - 2.0/π,
                       casimir_plateau = π/24,
                       two_sided_FSS   = false)
    δed    = res.δenergy_density
    Ns     = res.Ns
    nN, nD = size(δed)
    @assert size(res.FSS_filter_v2) == (nN, nD) "FSS_filter_v2 size mismatch with δenergy_density"

    # bulk density perturbation = |N²·(δe₀ − boundary_e0/N)|  (= res.density_pert for default e0)
    dens_pert(iN, iD) = abs((δed[iN, iD] - boundary_e0 / Ns[iN]) * Ns[iN]^2)

    log_cas     = log(casimir_plateau)
    log_tol_FSS = log(1 + fss_tol)

    fill!(res.FSS_filter_v2, false)
    @inbounds for iN in 1:nN, iD in 1:nD
        dp = dens_pert(iN, iD)
        (isfinite(dp) && dp > 0) || continue
        Δ = log(dp) - log_cas
        is_fss = two_sided_FSS ? (abs(Δ) < log_tol_FSS) : (Δ < log_tol_FSS)
        is_fss && (res.FSS_filter_v2[iN, iD] = true)
    end

    return res
end


# Source Pluto cell: cece56c6-c483-4194-945f-959a3464801b
const CLASSIFY_SETTINGS = Dict{Symbol, NamedTuple}(
:TFI_only    => (
	init_kappa = 1.0,
	fss_tol = 0.08,
	D_fit = 2:12,
	casimir_label_FDS = :below
),

:Rxx_Rz      => (
	init_kappa = 0.5,
	fss_tol = 0.08,
	D_fit = D_scan[1:end],
	casimir_label_FDS = :below
),

:kitaev_only => (
	init_kappa = 3/4,
	fss_tol = 0.08,
	D_fit = 3:12,
	casimir_label_FDS = :above
),

:kitaev_HVA  => (
	init_kappa = 1.0,
	fss_tol = 0.08,
	D_fit = 2:12,
	casimir_label_FDS = :above
),

:TFI_EXP     => (
	init_kappa = 2.5,
	fss_tol = 0.08,
	D_fit = 2:12,
	casimir_label_FDS = :below
),

:EXP_HVA     => (
	init_kappa = 2.0,
	fss_tol = 0.08,
	D_fit = 1:12,
	casimir_label_FDS = :above
),

:TFI_POW     => (
	init_kappa = 1.0,
	fss_tol = 0.08,
	D_fit = 2:12,
	casimir_label_FDS = :below
),

:POW_HVA     => (
	init_kappa = 3/4,
	fss_tol = 0.08,
	D_fit = 2:12,
	casimir_label_FDS = :below
),
)


# Source Pluto cell: 123d7588-304a-497a-909c-bdd93cb9fc3e
# Y has shape (iN, iD): first axis L, second axis D
function grid_to_long(Lvals, Dvals, Y)
    @assert size(Y) == (length(Lvals), length(Dvals))
    L = repeat(Lvals, outer=length(Dvals))      # column-major: L varies fastest
    D = repeat(Dvals, inner=length(Lvals))
    y = vec(Y)                                   # also column-major
    return L, D, y
end


# Source Pluto cell: 83a0bef3-996d-4843-a045-19a3d1c238c0
# --- envelope: peak (max) of lr within each width-w r-bin ---
function envelope_peaks(r, lr; env_window=3)
    rmin = first(r)
    bins = unique(@. (r - rmin) ÷ env_window)
    pk = map(bins) do b
        m = @. (r - rmin) ÷ env_window == b
        j = argmax(@view lr[m])
        (@view(r[m])[j], @view(lr[m])[j])
    end
    return first.(pk), last.(pk)
end


# Source Pluto cell: 38038c19-9371-4d60-8083-9bf73756a3c7
function loglinear_huber(x, z; huber_polish=true)
    X = hcat(x, ones(length(x))); w = ones(length(x))     # note: [slope, intercept] order
    β = (sqrt.(w).*X) \ (sqrt.(w).*z)
    if huber_polish
        for _ in 1:20
            r = z .- X*β
            s = 1.4826*median(abs.(r .- median(r))) + eps(); δ = 1.345*s
            w = @. ifelse(abs(r) ≤ δ, 1.0, δ/abs(r))
            β = (sqrt.(w).*X) \ (sqrt.(w).*z)
        end
    end
    # coefficient covariance at the converged weights
    r   = z .- X*β
    dof = max(length(x) - 2, 1)
    s²  = sum(w .* r.^2) / dof
    C   = s² .* inv(X' * (w .* X))          # 2×2 cov of [slope, intercept]
    se_slope     = sqrt(max(C[1,1], 0))
    se_intercept = sqrt(max(C[2,2], 0))
    return β, (se_slope=se_slope, se_intercept=se_intercept)
end


# Source Pluto cell: ca33b9a4-1d75-4cc1-9d01-a814cd665229
function _xi_from_points(r, lr; env_window, min_env, huber_polish)
    length(r) ≥ min_env || return (ξ=NaN, A=NaN, n=length(r), env_n=0,
                                   ξ_relerr=NaN, A_se=NaN)
    re, le = envelope_peaks(r, lr; env_window)
    length(re) ≥ min_env || return (ξ=NaN, A=NaN, n=length(r), env_n=length(re),
                                    ξ_relerr=NaN, A_se=NaN)
    β, se = loglinear_huber(re, le; huber_polish)
    ξ = β[1] < 0 ? -inv(β[1]) : NaN
    A = exp(β[2])
    # ξ = -1/slope ⇒ relative error of ξ equals relative error of slope
    ξ_relerr = (isfinite(ξ) && β[1] != 0) ? abs(se.se_slope / β[1]) : NaN
    # A = exp(intercept) ⇒ se_A = A · se_intercept  (and log-space error = se_intercept)
    A_se     = A * se.se_intercept
    return (ξ=ξ, A=A, slope=β[1], intercept=β[2],
            ξ_relerr=ξ_relerr, ξ_se=(isfinite(ξ) ? ξ*ξ_relerr : NaN),
            A_se=A_se, lnA_se=se.se_intercept,
            n=length(r), env_n=length(re), r=r, lr=lr, re=re, le=le)
end


# Source Pluto cell: 587832f3-d2e6-4224-bc18-a30a8b333709
function fit_xi_envelope_cft(Cd, N, Δ; rfrac=(0.05,0.45), ref_site=0,
                             env_window=3, min_env=4, huber_polish=true, log_floor=-32.0)
    rlo, rhi = round(Int, rfrac[1]*N), round(Int, rfrac[2]*N)
    chord(rr) = (N/π) * sin(π*rr/N)
    # rr
    pts = [(rr, lc + 2Δ*log(chord.(rr)))
           for rr in rlo:rhi
           for c in (abs(Cd[ref_site+rr]),)              # bind c once
           for lc in (log(abs(c)),)                 # bind lc = ln|c|, visible below
           if isfinite(c) && c != 0 && lc >= log_floor]
    _xi_from_points(first.(pts), last.(pts);
                    env_window, min_env, huber_polish)
end


# Source Pluto cell: 46a959d5-d8d7-43d4-a466-d5f31f098415
# D-jackknife on the correlator power-law fit: leave out one D, refit κ, take the spread.
# ξs, seln are the per-D ξ and its log-space error (ξ_relerr), as fed to the weighted OLS.
function kappa_corr_jackknife_D(Dvals, ξs, seln)
    # Dvals, ξs, seln aligned; use only finite, positive, weighted points
    ok = [isfinite(ξs[i]) && ξs[i] > 0 && isfinite(seln[i]) && seln[i] > 0
          for i in eachindex(ξs)]
    D = Float64.(Dvals[ok]); z = log.(ξs[ok]); w = 1 ./ seln[ok].^2
    n = length(D)
    n ≥ 4 || return (κ_jack=NaN, se_jack=NaN, κs=Float64[], n_jack=0)

    fitκ(idx) = let X = hcat(log.(D[idx]), ones(length(idx))), sw = sqrt.(w[idx])
        ((sw .* X) \ (sw .* z[idx]))[1]        # weighted-LS slope = κ
    end

    κs = [fitκ(setdiff(1:n, i)) for i in 1:n]   # leave-one-D-out
    κ̄  = sum(κs)/n
    se_jack = sqrt((n-1)/n * sum((κs .- κ̄).^2))
    return (κ_jack=κ̄, se_jack=se_jack, κs=κs, n_jack=n)
end


##%
# Correlation-length fits
# Source Pluto cell: d227b029-4ff8-4ff8-8b71-365323a45f65
CLASSIFY_SETTINGS_CORR = Dict(
	:TFI_only  => (N_best = 512, D_fit = [2:9...,], logfloor=-32),
	:Rxx_Rz  => (N_best = 512, D_fit = 1:11, logfloor=-60),
	:kitaev_only  => (N_best = 512, D_fit = 2:7, logfloor=-32),
	:kitaev_HVA  => (N_best = 512, D_fit = [3,4,5,6,7,8,9,10,11,12], logfloor=-34),
	:EXP_HVA => (N_best = 512, D_fit = [2,3,4,5,6,7,8,9], logfloor=-35),
	:TFI_EXP => (N_best = 512, D_fit = [2,3,5,6,7,8,9], logfloor=-32),
	:TFI_POW => (N_best = 256, D_fit = [2,3,4,5,6,7,8,9], logfloor=-32),
	:POW_HVA => (N_best = 128, D_fit = [1,2,3,4,], logfloor=-32),
)


# Source Pluto cell: 11da40bc-c2ee-43ed-a2b5-df6461e93046
N_fit_xi_corr = [60,64,128,256,512]


# Source Pluto cell: d3e8f00e-700f-4df6-b9a9-b2c1d4929541
begin
	
	xi_corr_by_D = Dict(
		[
			key => [
				[
					fit_xi_envelope_cft(
						results[key].correlator[N_test,D],
						N_test, 1/8;
						log_floor=CLASSIFY_SETTINGS_CORR[key].logfloor,
						rfrac=(0.05,0.4),
						env_window = D
					)
					for D in 1:12
				]
				for N_test in N_fit_xi_corr
			]
			for key in loop_keys
		]
	)

	xi_vs_D_corr = Dict{Symbol,NamedTuple}()

    for key in loop_keys
        Dgood = CLASSIFY_SETTINGS_CORR[key].D_fit
        N     = CLASSIFY_SETTINGS_CORR[key].N_best
        iN    = findfirst(x -> x == N, N_fit_xi_corr)

        ξs   = [xi_corr_by_D[key][iN][D].ξ        for D in 1:12]
        seln = [xi_corr_by_D[key][iN][D].ξ_relerr for D in 1:12]   # se on log ξ

        ok = [D in Dgood && isfinite(ξs[D]) && ξs[D] > 0 && isfinite(seln[D]) && seln[D] > 0
              for D in 1:12]
        Duse = findall(ok)
        @assert length(Duse) ≥ 3 "$key: fewer than 3 usable ξ points"

        z  = log.(ξs[Duse])
        X  = hcat(log.(Float64.(Duse)), ones(length(Duse)))
        w  = 1 ./ seln[Duse].^2              # inverse-variance in log space
        sw = sqrt.(w)

        β  = (sw .* X) \ (sw .* z)           # weighted LS: [κ, ln A]
        κ_corr, A_corr = β[1], exp(β[2])

        n   = length(z); dof = max(n - 2, 1)
        r   = z .- X*β
        s²  = sum(w .* r.^2) / dof           # reduced χ² (rescales cov to actual scatter)
        C   = s² .* inv(X' * (w .* X))       # cov of [κ, ln A]

        se_κ   = sqrt(max(C[1,1], 0))
        se_lnA = sqrt(max(C[2,2], 0))
        se_A   = A_corr * se_lnA
		
		jkc = kappa_corr_jackknife_D(Duse, ξs[Duse], seln[Duse])

        xi_vs_D_corr[key] = (; κ_corr, A_corr,
                       κ_se = se_κ, A_se = se_A, lnA_se = se_lnA,
                       cov_κ_lnA = C[1,2], χ²_red = s², n_fit = n,
                       κ_se_jack = jkc.se_jack,   
                       κ_jack    = jkc.κ_jack,
                       A_lo = A_corr*exp(-se_lnA), A_hi = A_corr*exp(+se_lnA))
    end
	


	


end


##%
# Energy-scaling fits
# Source Pluto cell: 759a7176-b156-4173-a11a-2c42d50d17ad
# Free κ (VARPRO outer), bounded Q (constrained inner). κ seeded from correlator.
function fit_energy_2d_freeκ_boundedQ(L, D, y, κ_seed, A;
                                      Dmin_for_bound=nothing, κ_window=0.75,
                                      weights=:relative, nQ=151,
                    )
    L=float.(collect(L)); D=float.(collect(D)); y=float.(collect(y))
    keep = isfinite.(L) .& isfinite.(D) .& isfinite.(y)
    L,D,y = L[keep],D[keep],y[keep]
    lnD = log.(D)
    w = weights===:relative ? 1 ./(y.^2 .+ eps()) : ones(length(y))
    Dmin = something(Dmin_for_bound, minimum(D))

    # inner: at fixed κ, scan Q over its physical box, P projected linearly
    function inner(κ)
        base = L.^2 .* D.^(-2κ)
        Qmin, Qmax = -κ*log(Dmin), log(A)
        Qs = range(Qmin, Qmax; length=nQ)
        best_ssr = Inf; best = (Q=NaN, P=NaN)
        for Q in Qs
            φ = base .* (κ .* lnD .+ Q)
            s = sum(w .* φ.^2)
            s > 0 || continue
            P = sum(w .* φ .* y) / s
            ssr = sum(w .* (y .- P.*φ).^2)
            ssr < best_ssr && (best_ssr = ssr; best = (Q=Q, P=P))
        end
        return best_ssr, best
    end

    # outer: 1D search over κ, seeded at correlator value, bracketed ±window
    lo, hi = κ_seed*(1-κ_window), κ_seed*(1+κ_window)
    res = Optim.optimize(κ -> first(inner(κ)), lo, hi, Optim.Brent())          # Brent on bracket
    κ̂ = Optim.minimizer(res)
    ssr̂, b = inner(κ̂)

    # κ error from curvature of the (Q-profiled) SSR(κ)
    h = 1e-3*max(κ̂,1e-2)
    d2 = (first(inner(κ̂+h)) - 2ssr̂ + first(inner(κ̂-h)))/h^2
    dof = max(length(y)-3, 1); σ² = ssr̂/dof
    κ_se = d2 > 0 ? sqrt(2σ²/d2) : NaN

    return (κ=κ̂, κ_se=κ_se, Q=b.Q, P=b.P, a=A*exp(-b.Q),
            ssr=ssr̂, n_used=length(y), κ_seed=κ_seed, A=A,
            κ_bracket=(lo,hi), hit_bound = κ̂ ≤ lo+1e-6 || κ̂ ≥ hi-1e-6)
end


# Source Pluto cell: a6152f29-6349-42bf-a92e-690720ff441c
# D-column jackknife: refit leaving out one D at a time; spread of the leave-one-out
# κ's = the error. Exposes single-point-D leverage that the curvature κ_se hides.
function kappa_jackknife_D(L, D, y, κ_seed, A; Dmin_for_bound, κ_window=0.5)
    finite = isfinite.(L) .& isfinite.(D) .& isfinite.(y)
    Ds = sort(unique(D[finite]))
    κs = Float64[]
    for dleave in Ds
        m = D .!= dleave
        count(finite .& m) ≥ 4 || continue          # need enough points to refit
        fb = fit_energy_2d_freeκ_boundedQ(L[m], D[m], y[m], κ_seed, A;
                                          Dmin_for_bound, κ_window)
        (isfinite(fb.κ) && !fb.hit_bound) && push!(κs, fb.κ)
    end
    n = length(κs)
    n ≥ 2 || return (κ_jack=NaN, se_jack=NaN, κs=κs, n_jack=n)
    κ̄ = sum(κs)/n
    se_jack = sqrt((n-1)/n * sum((κs .- κ̄).^2))      # jackknife standard error
    return (κ_jack=κ̄, se_jack=se_jack, κs=κs, n_jack=n)
end


# Source Pluto cell: 39f4e5c4-fb25-48fd-bd83-f1f47d8884b4
begin
    κs_v4 = Dict{Symbol, NamedTuple}()

    for key in loop_keys
        haskey(results, key)             || (println("skip $key: no results loaded"); continue)
        haskey(CLASSIFY_SETTINGS, key)   || (println("skip $key: no energy classify settings"); continue)
        haskey(xi_vs_D_corr, key)        || (println("skip $key: no correlator κ/A"); continue)

        s          = CLASSIFY_SETTINGS[key]
        κ_corr     = xi_vs_D_corr[key].κ_corr
        A_corr     = xi_vs_D_corr[key].A_corr
        D_fit      = s.D_fit
        D_fit_inds = map(dd -> findfirst(==(dd), D_scan), D_fit)

        # 1) tag the FSS regime in place → results[key].FSS_filter_v2
        classify_FSS!(results[key];
            fss_tol         = 0.5,
            boundary_e0     = 1.0 - 2.0/π,
            casimir_plateau = π/24,
        )

        # 2) keep everything OUTSIDE the FSS regime
        keep        = .!results[key].FSS_filter_v2
        pert_masked = ifelse.(keep, results[key].pert_data, NaN)

        # 3) long-format (NaN-filtered downstream)
        L_long, D_long, y_long =
            grid_to_long(N_scan, D_fit, pert_masked[:, D_fit_inds])

        Dmin_bound = minimum(CLASSIFY_SETTINGS_CORR[key].D_fit)

        # 4) bounded-Q, seeded-κ energy fit (returns curvature κ_se)
        fb = fit_energy_2d_freeκ_boundedQ(
            L_long, D_long, y_long,
            κ_corr, A_corr;
            Dmin_for_bound = Dmin_bound,
            κ_window = 0.5,
        )

        # 5) D-jackknife error (feels single-point-D leverage; widens sparse rows)
        jk = kappa_jackknife_D(
            L_long, D_long, y_long,
            κ_corr, A_corr;
            Dmin_for_bound = Dmin_bound,
            κ_window = 0.5,
        )

        κs_v4[key] = merge(fb, (
            κ_corr  = κ_corr,
            se_jack = jk.se_jack,      # jackknife κ error (report this)
            κ_jack  = jk.κ_jack,       # jackknife mean κ (bias check vs fb.κ)
            n_jack  = jk.n_jack,       # how many leave-one-out refits contributed
        ))
    end
end; md"Energy fit (κ seeded, Q bounded) + D-jackknife κ error. Report `se_jack`; compare against curvature `κ_se`."


##%
# Central findings table
# Source Pluto cell: ce288003-f73d-4321-bcd1-8ad8278f83b8
"""
    central_findings_table(; min_pts=3) -> Markdown.MD

Build the publication table containing the central Ising scaling results.
"""
function central_findings_table(; min_pts = 3)

    fit_coverage(key) = let
        Dinds   = [findfirst(==(d), D_scan) for d in CLASSIFY_SETTINGS[key].D_fit]
        keepcol = falses(size(results[key].pert_data, 2)); keepcol[Dinds] .= true
        mask = isfinite.(results[key].pert_data) .& .!results[key].FSS_filter_v2 .&
               reshape(keepcol, 1, :)
        # per-D surviving point counts
        ppD  = [(D_scan[iD], count(@view mask[:, iD])) for iD in axes(mask,2)
                if any(@view mask[:, iD])]
        Dal  = first.(ppD)
        # thresholded: only D with ≥ min_pts points
        Dstrong = [D for (D,n) in ppD if n ≥ min_pts]
        (; n_D       = length(Dal),
           D_range   = isempty(Dal)     ? "–" : "$(minimum(Dal))–$(maximum(Dal))",
           n_D3      = length(Dstrong),
           D_range3  = isempty(Dstrong) ? "–" : "$(minimum(Dstrong))–$(maximum(Dstrong))",
           lnD_span3 = length(Dstrong) < 2 ? 0.0 : log(maximum(Dstrong)) - log(minimum(Dstrong)),
           pts_per_D = ppD)
    end

    rows = join([
        let κ_corr   = xi_vs_D_corr[key].κ_corr,
            κ_cor_se = xi_vs_D_corr[key].κ_se_jack,
            κ_varpro = κs_v4[key].κ,
            κ_varpro_se = κs_v4[key].se_jack,
            cov      = fit_coverage(key)

            "| $(string(VARIANTS[key].label)) | $(round(κ_corr; sigdigits=3)) ± $(round(κ_cor_se,sigdigits=2)) | " *
            "$(round(κ_varpro; sigdigits=3)) ± $(round(κ_varpro_se,sigdigits=2)) | " *
            "$(cov.n_D) | $(cov.D_range) | " *
            "$(cov.n_D3) | $(cov.D_range3) |"
        end
        for key in [
            :Rxx_Rz, :TFI_only, :kitaev_HVA, :kitaev_only,
            :EXP_HVA, :TFI_EXP, :POW_HVA, :TFI_POW,
        ]
    ], "\n")

    Markdown.parse("""
    #### Ising model
    **Mean** Scaling exponents κ from bulk energy density and running coupling collapse of data **outside the FSS regime**. The last two columns count depths with ≥ $(min_pts) surviving system sizes.

    | Ansatz | κ : C_{XX} | κ : g(L/ξ) varpro | n_D | D range | n_D(≥$(min_pts)) | D range (≥$(min_pts)) |
    |:-------|--------------------:|-----------------------------:|-------:|:----------:|------------------------:|:-------------------------------:|
    $(rows)
    """)
end

central_findings_table()

##%
# Shared plotting styles
# Source Pluto cell: cb543e7a-2937-4d86-9cad-99309aed7d9a
begin 
	ms = [:circle, :rect, :diamond, :rtriangle, :ltriangle, :vline, :hline, :utriangle, :dtriangle, :pentagon, :cross, :xcross, :star4,  :star5, :star6, :star8, :hexagon, :octagon]
	markers       = [ms[ii%length(ms)+1] for ii in 0:length(N_scan)-1]

	color_ref = :royalblue2 # crimson RoyalBlue red4 firebrick3 cornflowerblue
end; md" define markers and colours to plot"


# Source Pluto cell: 81bcad58-667d-447d-9227-51770d868697
cmap_L = reverse(cgrad(:bamako, 255, categorical = true)[1:end-35])


# Source Pluto cell: 9a8d7fa8-a98c-47e4-81f4-769b946b06d6
cmap_D = cgrad(:batlow, 255, rev=true) # magma Reds
# cmap_D = cgrad(:batlowK, rev = true)


##%
# Correlation-length figure
# Source Pluto cell: a0fec655-d5ee-462e-8c4e-fdd4c0b4f93c
"""
    plot_xi_corr_panel(key; kwargs...) -> Figure

Two-panel correlation-length figure for one ansatz `key`:
(a) chord-corrected ratio decay ln[C_XX·d_chord^η] vs u = r/L with exponential fits,
(b) ξ(D) vs D with the ξ = A·D^κ power-law fit.
"""
function plot_xi_corr_panel(key;
        Ds            = 1:12,
        settings      = CLASSIFY_SETTINGS_CORR,
        xi_by_D       = xi_corr_by_D,
        xi_vs_D       = xi_vs_D_corr,
        N_grid        = N_fit_xi_corr,
        variants      = VARIANTS,
        cmap          = cmap_D,
        size          = (320, 340))

    N      = settings[key].N_best
    Dgood  = settings[key].D_fit
    iN     = findfirst(==(N), N_grid)

    f    = Figure(figure_padding = 4, size = size)
    grid = f[1,1] = GridLayout()

    # ---- (a) ln|C_D · chord^η| vs u = r/L ----
    axD = Axis(grid[1,1], xlabel = "𝑢 = 𝑟/𝐿",
               ylabel = rich("ln[ 𝐶", subscript("𝑋𝑋"), "(𝑟) · 𝑑",subscript("chord"),"(𝑟,𝐿)", superscript("1/4"), " ]"),
               title = "$(variants[key].label), 𝐿 = $N", titlefont = :regular,
               xticks = 0.1:0.1:0.5,
               xminorticks = IntervalsBetween(10),
               xlabelpadding = 0)

    ξs = fill(NaN, length(Ds))
    for (cD, D) in enumerate(Ds)
        fit = xi_by_D[key][iN][D]
        isnan(fit.A) && continue
        scatter!(axD, fit.r ./ N, fit.lr; color = cD, colormap = cmap,
                 colorrange = (1, length(Ds)), markersize = 3.5, label = "𝐷=$D")
        if isfinite(fit.ξ)
            rr = range(minimum(fit.r), maximum(fit.r); length = 2) ./ N
            lines!(axD, rr, fit.slope .* rr .* N .+ fit.intercept;
                   color = cD, colormap = cmap, colorrange = (1, length(Ds)),
                      # linestyle=(:dot, :dense),
                  )
            ξs[cD] = fit.ξ
        end
    end

    # ---- (b) ξ(D) vs D with power-law fit ----
    axX = Axis(grid[2,1:2], xlabel = "𝐷", ylabel = rich("𝜉",subscript("𝐷")),
               xscale = log10, yscale = log10,
               xticks = (1:12, [string.(1:10)..., "", ""]),
               xminorticksvisible = false, xminorgridvisible = false)
    isgood =  map( dd -> in(dd,Dgood), Ds)
    scatter!(axX, Ds[isgood], ξs[isgood]; markersize = 9, strokewidth=1, color=Makie.wong_colors()[1])
    scatter!(axX, Ds[.!isgood], ξs[.!isgood]; markersize = 9, strokewidth=0, color=Makie.wong_colors()[1])
    xlims!(axX, 0.94, 13)

     κ_corr = A_corr = NaN
    if length(Dgood) ≥ 3
        v = xi_vs_D_corr[key]
        κ_corr, A_corr = v.κ_corr, v.A_corr

        # covariance from the fit (for the intercept and κ–lnA correlation) ...
        C = [v.κ_se^2      v.cov_κ_lnA
             v.cov_κ_lnA   v.lnA_se^2]
        # ... but rescale the κ variance to the (wider) jackknife error, keeping the
        # κ–lnA correlation coefficient fixed so the band stays a valid covariance.
        κ_se_use = hasproperty(v, :κ_se_jack) && isfinite(v.κ_se_jack) ? v.κ_se_jack : v.κ_se
        ρ  = v.cov_κ_lnA / (v.κ_se * v.lnA_se)              # correlation coefficient
        C_band = [κ_se_use^2                ρ*κ_se_use*v.lnA_se
                  ρ*κ_se_use*v.lnA_se       v.lnA_se^2]

        A_se  = A_corr * v.lnA_se
        Dpr   = logrange(minimum(Dgood), maximum(Dgood), 100)
        logξ  = κ_corr .* log.(Dpr) .+ log(A_corr)
        selog = [ sqrt(max([log(D),1.0]' * C_band * [log(D),1.0], 0)) for D in Dpr ]
        ξ_hat, ξ_lo, ξ_hi = exp.(logξ), exp.(logξ .- selog), exp.(logξ .+ selog)

        band!(axX, Dpr, ξ_lo, ξ_hi; color = (:crimson, 0.22))
        lines!(axX, Dpr, ξ_hat; color = :crimson,
               label = rich("fit: ($(round(A_corr; sigdigits=3)) ± $(round(A_se; sigdigits=2)))⋅𝐷",
                            superscript("($(round(κ_corr; sigdigits=3)) ± $(round(κ_se_use; sigdigits=2)))")))
        axislegend(axX, position = :lt, padding = (2,2,-1,-1), framevisible=false,)
    end

    # ---- discrete D colorbar ----
    let _cmap_D = cgrad(cmap, length(Ds), categorical = true)
        Colorbar(grid[1,2], colormap = _cmap_D,
                 limits = (0.5, length(Ds)+0.5),
                 ticks  = (1:length(Ds), string.(Ds)),
                 label  = "𝐷", labelpadding = 2, labelsize = 11,
                 tellheight = true, vertical = true)
    end

    colgap!(grid, 1, 3)
    rowgap!(grid, 6)
    Label(grid[1,1,TopLeft()], "(a)", font = :bold, halign = :left, valign=:top, fontsize = 10, padding=(-2,0,0,0))
    Label(grid[2,1,TopLeft()], "(b)", font = :bold, halign = :left, valign=:top, fontsize = 10, padding=(-2,0,0,0))

    return f
end


##%
# Vertical Ising figure
# Source Pluto cell: b91816db-1db1-4af3-9ced-438d1a5bc5b5
function make_figure_vert_Ising(res::CircuitResults;
                                  κ,
                                  A_xi    = 1.0,
                                  a_uv    = 1.0,
                                  marginal = true,
                                  FSS_filter        = res.FSS_filter_v2,
                                  D_plot            = D_scan,
                                  N_scan            = N_scan,
                                  D_scan            = D_scan,
                                  D_fit             = D_scan[2:12],
                                  ref_D_run_coupling = 3,
                                  ref_N_run_coupling = 10,
                                  fss_tol           = 0.5,
                                  distance          = 0.15,
                                  x_upper           = Inf,
                                  casimir_plateau   = π/24,
                                  casimir_label     = "𝜋𝑣𝑐/24",
                                    casimir_label_FDS = :above,
                                  boundary_e0       = 1.0 - 2.0/π,
                                  η_corr            = 0.25,
                                  corr_index        = (N, dist) -> round(Int, N*dist),
                                  alpha_plot        = 0.7,
                                  N_cap_FDS         = length(N_scan),
                                  y_cut = 1e-6)

    η_label = isinteger(η_corr) ? string(Int(η_corr)) :
              η_corr == 0.25 ? "1/4" : η_corr == 0.5 ? "1/2" : string(η_corr)
    κ_print = round(κ; digits=3) == 0.5 ? "1/2" : round(κ; digits=3)
    Q = log(A_xi / a_uv)

    logfac_xi(D) = κ*log(D) + Q

    ξ(D) = A_xi * D^κ

    f    = Figure(size=(320, 580), padding=1, fontsize=11)
    grid = f[1,1] = GridLayout()

    scatter_kw(cD) = (color = cD, colormap = cmap_D, colorrange = (1, length(D_plot)))

    inset_kw = (tellheight=false, tellwidth=false, backgroundcolor=:white,
        xlabelsize=9, ylabelsize=9, ylabelpadding=0, xlabelpadding=0,
        xticklabelsize=8, yticklabelsize=8)

    let _cmap_D = cgrad(cmap_D, length(D_plot), categorical=true)
        Colorbar(grid[0, 1:2], colormap=_cmap_D, limits=(0.5, length(D_plot)+0.5),
            ticks=(1:length(D_plot), string.(D_plot)),
            label="𝐷"; labelpadding=2, labelsize=11, tellheight=true, vertical=false)
    end

    Label(grid[1,1, Left()], "(a)", halign=:left, valign=:top, font=:bold, padding=(-2,0,0,0))
    Label(grid[2,1, Left()], "(b)", halign=:left, valign=:top, font=:bold, padding=(-2,0,0,0))
    Label(grid[2,2, Left()], "(c)", halign=:left, valign=:top, font=:bold, padding=(-2,0,0,0))
    Label(grid[3,1, Left()], "(d)", halign=:left, valign=:top, font=:bold, padding=(-2,0,0,0))

    correlator_dict = res.correlator

    # ════════════════════════════════════════════════════════════════════
    # (a) δE₀ vs L  (taller)  +  inset: finite-SIZE bulk plateau (Casimir)
    # ════════════════════════════════════════════════════════════════════
    axE0 = Axis(grid[1, 1:2], height = 180,
        xlabel = "𝐿", ylabel = rich("𝛿𝐸", subscript("abs")),
        xscale = log2, yscale = log10,
        yticks = (10.0.^(-16:2:2), [rich("10", superscript("$(sign(x)==-1 ? "−" : "")$(abs(x))")) for x in -16:2:2]),
        xticks = (2.0.^(0:1:10), [rich("2", superscript("$(x)")) for x in 0:1:10]))
    for (cD, iD) in enumerate(D_plot)
        sel_fss = FSS_filter[:, cD];  sel_co = .!sel_fss
        xs = N_scan
        ys = res.δE0[:,cD]
        
        # scatter!(axE0, N_scan, [1:length(N_scan), cD]; scatter_kw(cD)..., marker = markers)
        scatter!(axE0, xs[sel_co],  ys[sel_co];  scatter_kw(cD)..., marker = markers[sel_co],  alpha = alpha_plot)
        scatter!(axE0, xs[sel_fss], ys[sel_fss]; scatter_kw(cD)..., marker = markers[sel_fss], alpha = alpha_plot, strokewidth=0.8)
    end
    ylims!(axE0, 0.1*minimum(filter(!iszero,(filter(!isnan, res.δE0[1, 1:length(D_plot)])))),
                 1.5  *maximum(filter(!isnan, res.δE0[:, D_plot[1]])))

    # inset of (a): finite-SIZE plateau  L²|δe_bulk| vs L/D^κ  with Casimir line
    ax_FSS = Axis(grid[1, 1:2]; halign=0.99, valign=0.27,
        width=Relative(0.4), height=Relative(0.4),
        xscale = log2, yscale = log10,
        xlabel = rich("𝐿 / 𝐷", superscript("$(κ_print)")),
        ylabel = rich("𝐿", superscript("2"), " |𝛿𝑒", subscript("bulk"),"|"),
        xticks = (10.0.^(-4:1:3), [rich("10", superscript("$(sign(x)==-1 ? "−" : "")$(abs(x))")) for x in -4:1:3]),
                  xminorticks = IntervalsBetween(9),
        yticks = (10.0.^(-10:2:4), [rich("10", superscript("$(sign(x)==-1 ? "−" : "")$(abs(x))")) for x in -10:2:4]),
        inset_kw...)
    y_lo_cut, y_hi_cut = 1e-3, 1e1
    x_lo_I, x_hi_I = Inf, -Inf
    for (cD, D) in enumerate(D_plot)
        iD = findfirst(==(D), D_scan); isnothing(iD) && continue
        xs  = N_scan ./ ξ(D)
        ys2 = res.density_pert[:, iD]
        sel_fss = FSS_filter[:, iD];  sel_co = .!sel_fss
        scatter!(ax_FSS, xs[sel_co],  ys2[sel_co];  scatter_kw(cD)..., marker = markers[sel_co],  alpha = alpha_plot)
        scatter!(ax_FSS, xs[sel_fss], ys2[sel_fss]; scatter_kw(cD)..., marker = markers[sel_fss], alpha = alpha_plot, strokewidth=0.8)
        for (x, y) in zip(xs, ys2)
            (isfinite(x) && isfinite(y) && y_lo_cut ≤ y ≤ y_hi_cut) || continue
            x_lo_I = min(x_lo_I, x); x_hi_I = max(x_hi_I, x)
        end
    end
    hlines!(ax_FSS, [casimir_plateau]; color=color_ref)
    hlines!(ax_FSS, [casimir_plateau * (1 + fss_tol)]; color=:gray, linestyle=(:dash, :dense))
    ylims!(ax_FSS, y_lo_cut, y_hi_cut)
    xlower = min(0.7*x_lo_I, 0.1)
    xhigher = max(1.4*x_hi_I, 10,)
    x_lo_I < x_hi_I && xlims!(ax_FSS, xlower, xhigher)
    tt = text!(ax_FSS, 0.9xhigher, 0.8casimir_plateau; text = casimir_label, color=color_ref, align = (:right,:top) )
    translate!(ax_FSS.blockscene, 0, 0, 150)

    # ════════════════════════════════════════════════════════════════════
    # (b) crossover  L·δE / ln(ξ/a)  vs L  +  inset: finite-DEPTH rescaled plateau
    # ════════════════════════════════════════════════════════════════════
    axE = Axis(grid[2, 1],
               height= 120,
        ylabel = marginal ? rich("𝐿⋅𝛿𝐸", subscript("abs"), " / ln(𝜉/𝑎)")
                          : rich("𝐿⋅𝛿𝐸", subscript("abs")),
        xlabel = rich("𝐿 / 𝐷", superscript("$(κ_print)")),
        xscale = log10, yscale = log10,
        xticks = (10.0.^(0:1:4), [rich("10", superscript("$(x)")) for x in 0:1:4]),
        yticks = (10.0.^(-14:1:4), [rich("10", superscript("$(sign(x)==-1 ? "−" : "")$(abs(x))")) for x in -14:1:4]),
        xminorticks = IntervalsBetween(9), yminorticks = IntervalsBetween(9))

    offset2 = let
        x_ref = N_scan[ref_N_run_coupling] / D_scan[ref_D_run_coupling]^κ
        lf    = logfac_xi(D_scan[ref_D_run_coupling])
        y_raw = res.pert_data[ref_N_run_coupling, ref_D_run_coupling]
        y_ref = (isfinite(y_raw) && lf > 0) ? y_raw / lf : NaN
        (isfinite(y_ref) && y_ref > 0) ? y_ref / x_ref^2 : NaN
    end
        
    # y_lo_cut, y_hi_cut = 1e-3, 1e1
    x_lo_E, x_hi_E = Inf, -Inf
    y_lo_E, y_hi_E = Inf, -Inf
    for (cD, D) in enumerate(D_plot)
        D in D_fit || continue
        iD = findfirst(==(D), D_scan); isnothing(iD) && continue
        lf = logfac_xi(D);  lf > 0 || continue
        xs  = N_scan ./ ξ(D)
        ys     = res.pert_data[:, iD] ./ lf
        sel_co = .!FSS_filter[:, iD]
        scatter!(axE, xs[sel_co], ys[sel_co]; scatter_kw(cD)..., marker = markers[sel_co], alpha = alpha_plot)
        for (x, y) in zip(xs[sel_co], ys[sel_co])
            (isfinite(x) && isfinite(y) )|| continue
            x_lo_E = min(x_lo_E, x); x_hi_E = max(x_hi_E, x)
            y_lo_E = min(y_lo_E, y); y_hi_E = max(y_hi_E, y)
        end
    end
    let pr = logrange(x_lo_E, x_hi_E, 51)
            lines!(axE, pr, offset2 .* pr .^ 2; color = color_ref,
                   label = rich("∝ (𝐿/𝐷", superscript("$(κ_print)"), ")", superscript("2"))
            )
    end
    # axislegend(axE; position=:rb,padding=(3,3,0,0))
    text!(axE, x_hi_E,2y_lo_E; text=rich("∝ (𝐿/𝐷", superscript("$(κ_print)"), ")", superscript("2")), color = color_ref, align = (:right,:bottom))
    # ════════════════════════════════════════════════════════════════════
    # (c): finite-DEPTH plateau  D^{2κ}|δe_bulk|/ln(ξ/a)  vs D^κ/L 
    # ════════════════════════════════════════════════════════════════════
    ax_E_D = Axis(grid[2, 2];
                  # halign=0.92, valign=0.14,
        # width=Relative(0.46), height=Relative(0.5),
                  height= 120,
        ylabel = marginal ? rich("𝐷", superscript("2𝜅"), "|𝛿𝑒", subscript("bulk"), "|/ln(𝜉/𝑎)")
                          : rich("𝐷", superscript("2𝜅"), "|𝛿𝑒", subscript("bulk"), "|"),
        xlabel = rich("𝐷", superscript("$(κ_print)"), "/𝐿"),
        xscale = log10, yscale = log10,
        xticks = (10.0.^(-4:1:2), [rich("10", superscript("$(sign(x)==-1 ? "−" : "")$(abs(x))")) for x in -4:1:2]),
        # inset_kw...
        )
    x_lo_D, x_hi_D = Inf, -Inf;  y_lo_D, y_hi_D = Inf, -Inf
    for (cD, D) in enumerate(D_plot)
        iD = findfirst(==(D), D_scan); isnothing(iD) && continue
        lf = logfac_xi(D);  lf > 0 || continue
        xs = ξ(D) ./ N_scan
        ys = res.density_pert[:, iD] ./ (N_scan .^ 2) .* ξ(D)^2 ./ lf
        sel_fss = FSS_filter[:, iD];  sel_co = .!sel_fss
        scatter!(ax_E_D, xs[sel_co],  ys[sel_co];  scatter_kw(cD)..., marker = markers[sel_co],  alpha = alpha_plot)
        scatter!(ax_E_D, xs[sel_fss], ys[sel_fss]; scatter_kw(cD)..., marker = markers[sel_fss], alpha = alpha_plot, strokewidth = 0.8)
        for (x, y) in zip(xs, ys)
            (isfinite(y) && y > 0) || continue
            x_lo_D = min(x_lo_D, x); x_hi_D = max(x_hi_D, x)
            y_lo_D = min(y_lo_D, y); y_hi_D = max(y_hi_D, y)
        end
    end
    if x_lo_D < x_hi_D
        xlims!(ax_E_D, 0.7*x_lo_D, 1.4*x_hi_D);  ylims!(ax_E_D, 0.3*y_lo_D, 3*y_hi_D)
    end
    # hlines!(ax_E_D, [casimir_plateau]; color=color_ref)
    # # translate!(ax_E_D.blockscene, 0, 0, 150)
    # if casimir_label_FDS == :above
    #      text!(ax_E_D, 0.9x_lo_D, 2casimir_plateau; text = casimir_label, color=color_ref, align = (:left,:bottom) )
    # elseif casimir_label_FDS == :below
    #     text!(ax_E_D, 0.9x_lo_D, 0.5casimir_plateau; text = casimir_label, color=color_ref, align = (:left,:top) )
    # end

    # ════════════════════════════════════════════════════════════════════
    # (d) σσ correlator collapse  (own panel)
    # ════════════════════════════════════════════════════════════════════
    axCxx = Axis(grid[3, 1:2], 
                 # height = 100,
        ylabel = rich("𝐿", superscript(η_label), " |𝐶",subscript("𝑋𝑋"),"(𝑟)|"),
        xlabel = rich("𝐿/𝐷", superscript("$(κ_print)")),
        xscale = log10, yscale = log10,
        xticks = (10.0.^(-4:1:4), [rich("10", superscript("$(sign(x)==-1 ? "−" : "")$(abs(x))")) for x in -4:1:4]),
        xminorticks = IntervalsBetween(9), yminorticks = IntervalsBetween(9))
    x_lo_C, x_hi_C = Inf, -Inf
    for (cD, D) in enumerate(D_plot)
        D in D_fit || continue
        iD = findfirst(==(D), D_scan); isnothing(iD) && continue
        Ns_here = N_scan[:]
        N_eff_vec = Ns_here ./ ξ(D)
        Cxx_vec = [let idx_c = corr_index(N, distance)
                (haskey(correlator_dict,(N,D)) && 1 ≤ idx_c ≤ length(correlator_dict[(N,D)])) ?
                    correlator_dict[(N,D)][idx_c] : NaN end for N in Ns_here]
        ys = Ns_here .^ η_corr .* abs.(Cxx_vec)
        scatter!(axCxx, N_eff_vec, ys; scatter_kw(cD)..., marker = markers, alpha = alpha_plot)
        for (x, y) in zip(N_eff_vec, ys)
            (isfinite(x) && isfinite(y) && y > 0) || continue
            x_lo_C = min(x_lo_C, x);  y ≥ y_cut && (x_hi_C = max(x_hi_C, x))
        end
    end
    ylims!(axCxx, y_cut, 10)
    x_lo_C < x_hi_C && xlims!(axCxx, 0.75*x_lo_C, 1.25*x_hi_C)
    text!(axCxx, x_lo_C, 2y_cut; align=(:left,:bottom), text="𝑟 = $(round(distance,sigdigits=2)) 𝐿", fontsize=8)

    colgap!(grid, 8);  rowgap!(grid, 8)
    return f
end


##%
# Megaplot configuration and occupation helpers
# Source Pluto cell: 9330ff11-124b-4011-9ac9-506638d4b6fe
#### Per-ansatz plot settings
# Only N_cap_FDS and D_plot differ between ansätze; everything else is
# shared. Edit individual entries here without touching the figure code.
const PLOT_SETTINGS = Dict{Symbol, NamedTuple}(
    :Rxx_Rz      => (N_cap_FDS = length(N_scan),       D_plot = D_scan),
    :TFI_only    => (N_cap_FDS = length(N_scan) - 3,   D_plot = 1:12),
    :kitaev_only => (N_cap_FDS = length(N_scan),       D_plot = 1:12),
    :kitaev_HVA  => (N_cap_FDS = length(N_scan),       D_plot = 1:12),
    :TFI_EXP     => (N_cap_FDS = length(N_scan) - 1,   D_plot = 1:12),
    :EXP_HVA     => (N_cap_FDS = length(N_scan) - 1,   D_plot = 1:12),
    :TFI_POW     => (N_cap_FDS = length(N_scan) - 2,   D_plot = 1:12),
    :POW_HVA     => (N_cap_FDS = length(N_scan) - 2,   D_plot = 1:12),
)


# Source Pluto cell: 2d9023c1-8586-4ccf-90b0-0e0e4948f921
occupation_gs(N) = 	vcat(zeros(N), ones(N))


# Source Pluto cell: b416538a-6ce6-4247-84f0-f59f5209780e
psi_up(L::Int) = Hermitian(
    [
        diagm(ones(ComplexF64, L)) zeros(ComplexF64, L, L);
        zeros(ComplexF64, L, L) zeros(ComplexF64, L, L)
    ]
)


# Source Pluto cell: 7a2de8d5-c1e5-4524-8048-24d39a0b6b43
function occupation_up(N)
	HD, U = Fu.Diag_h(TFI_Hamiltonian(N,1,1), 0)
	return real(diag( U' * psi_up(N) * U))
end


# Source Pluto cell: 657d6966-9133-4dc4-9889-f626e44874f2
color_init = :crimson


##%
# Publication megaplot
# Source Pluto cell: 618f6f08-031f-4598-b414-7ca04db105cd
function make_mega_plot(plot_keys::Vector{Symbol};
                                    ref_D_run_coupling = 3,
                                    ref_N_run_coupling = 16,
                                    kappa_dict   = κs_v4,
                                    xi_vs_D       = xi_vs_D_corr,
                                    class_settings = CLASSIFY_SETTINGS,
                                    corr_settings  = CLASSIFY_SETTINGS_CORR,   # correlator classify
                                    xi_by_D        = xi_corr_by_D,             # per-D correlator fits
                                    N_grid_corr    = N_fit_xi_corr,            # correlator N grid
                                    alpha_plot   = 0.8,
                                    N_mode       = 64,
                                    D_show       = [1, 2, 3, 6, 12],
                                    column_width = 250,
                                    outer_width  = 60,
                                    page_width_mm = 170.0,
                                    page_height_mm = 257.0,
                                    caption_height_mm = 40.0,
                                    figure_fontsize = 14,
                                    top_fraction = 0.03,
                                    header_fraction = 0.015,
                                    row_weights = (
                                        ratio    = 1.10,
                                        xi       = 0.95,
                                        energy   = 1.00,
                                        plateaus = 0.9,
                                        collapse = 1.40,
                                        modes    = 1.25,
                                    ),
                                    band_alpha   = 0.22,
                                    distance     = 0.15,
                                    η_corr       = 0.25,
                                    casimir_plateau = π/24,
                                    casimir_label   = "𝜋/24",
                                    boundary_e0  = 1.0 - 2.0/π,
                                    y_cut        = 1e-5,
                                )
    η_label = isinteger(η_corr) ? string(Int(η_corr)) :
              η_corr == 0.25 ? "1/4" : η_corr == 0.5 ? "1/2" : string(η_corr)

    ncol = length(plot_keys)
    @assert ncol > 0 "make_mega_plot requires at least one ansatz"
    @assert 0 < caption_height_mm < page_height_mm
    @assert top_fraction ≥ 0 && header_fraction ≥ 0
    @assert top_fraction + header_fraction < 1

    # Preserve the printable-area aspect ratio after reserving vertical space
    # for the LaTeX caption. For four columns and the defaults this gives
    # 1060 × 1353, but both dimensions now adapt to `plot_keys` and the page.
    figwidth = round(Int, column_width * ncol + outer_width)
    usable_page_height_mm = page_height_mm - caption_height_mm
    figheight = round(Int, figwidth * usable_page_height_mm / page_width_mm)

    f = Figure(
        size = (figwidth, figheight),
        figure_padding = (2,7,2,-2),
        fontsize = figure_fontsize,
    )
    grid = f[1, 1] = GridLayout()

    # Rows -1 and 0 contain the depth bars and ansatz headings. The six data
    # rows share all remaining height according to normalized semantic weights.
    data_weights = Float64.(collect(values(row_weights)))
    @assert length(data_weights) == 6 "row_weights must contain six entries"
    @assert all(>(0), data_weights) "all row weights must be positive"
    data_fraction = 1 - top_fraction - header_fraction
    data_fractions = data_fraction .* data_weights ./ sum(data_weights)

    scatter_kw_N(iN) = (color = iN, colormap = cmap_L, colorrange = (1, length(N_scan)))

    inset_kw = (
        tellheight = false, tellwidth = false, backgroundcolor = :white,
        xlabelsize = round(Int, 0.7*figure_fontsize), ylabelsize = round(Int, 0.7*figure_fontsize), ylabelpadding = 0, xlabelpadding = 0,
        xticklabelsize = round(Int, 0.65*figure_fontsize), yticklabelsize = round(Int, 0.65*figure_fontsize),
    )

    mode_idx = vcat(1:N_mode, 2N_mode:-1:N_mode+1)

    for (j, plot_key) in enumerate(plot_keys)
        s          = PLOT_SETTINGS[plot_key]
        N_cap_FDS  = s.N_cap_FDS
        D_plot     = s.D_plot
        fss_tol    = 0.5
        κ          = kappa_dict[plot_key].κ
        key        = plot_key
        is_exp_ansatz = occursin("EXP", uppercase(String(key)))
        plateau_xtick_exponents = is_exp_ansatz ? (-2:2:2) : (-2:1:2)
        mode_xtick_exponents = is_exp_ansatz ? (-3:2:2) : (-3:1:2)
        FSS_filter = results[key].FSS_filter_v2
        A_xi = xi_vs_D[plot_key].A_corr
        ξ(D) = A_xi * D^κ
        a_uv = kappa_dict[key].a
        Q = log(A_xi / a_uv)

        logfac_xi(D) = κ*log(D) + Q

        let _cmap_D = cgrad(cmap_D, length(D_plot), categorical=true)
        Colorbar(grid[-1, j], colormap=_cmap_D, limits=(0.5, length(D_plot)+0.5),
            ticks=(1:length(D_plot), string.(D_plot)),
            label="𝐷"; labelpadding=2, labelsize=figure_fontsize, ticklabelsize=round(Int,0.8*figure_fontsize), tellheight=true, vertical=false)
        end
        
        
        offset2 = let
            x_ref = N_scan[ref_N_run_coupling] / ξ(ref_D_run_coupling)
            y_ref = results[key].pert_data[ref_N_run_coupling, ref_D_run_coupling] /logfac_xi(ref_D_run_coupling)
            isfinite(y_ref) && y_ref > 0 ? y_ref / x_ref^2 : NaN
        end
        κ_print = round(κ; sigdigits=3) == 0.5 ? "1/2" : round(κ; sigdigits=3)

        scatter_kw(cD)  = (color = cD, colormap = cmap_D, colorrange = (1, length(D_plot)))
        correlator_dict = results[key].correlator

        # correlator classification / grid for this ansatz
        N_corr     = corr_settings[key].N_best
        iN_corr    = findfirst(==(N_corr), N_grid_corr)
        Dgood_corr = corr_settings[key].D_fit

        # ══════════════════════════════════════════════════════════════════
        # Row 1: correlator chord-corrected ratio decay  ln[C·r^η] vs u = r/L
        # ══════════════════════════════════════════════════════════════════
        ax_ratio = Axis(grid[1, j],
            title = "𝐿 = $(CLASSIFY_SETTINGS_CORR[key].N_best)",
            titlefont = :regular,
            xlabel = "𝑢 = 𝑟/𝐿",
            ylabel = rich("ln[ 𝐶", subscript("𝑋𝑋"), "(𝑟)·𝑑",subscript("chord"),"(𝑟,𝐿)", superscript(η_label), " ]"),
            xticks = 0.1:0.1:0.5, xminorticks = IntervalsBetween(10), xlabelpadding = 0)
        if iN_corr !== nothing
            for (cD, D) in enumerate(1:12)
                haskey(xi_by_D, key) || continue
                fit = xi_by_D[key][iN_corr][D]
                isnan(fit.A) && continue
                scatter!(ax_ratio, fit.r ./ N_corr, fit.lr;
                         scatter_kw(cD)..., markersize = 3.5)
                if isfinite(fit.ξ)
                    rr = range(minimum(fit.r), maximum(fit.r); length = 2) ./ N_corr
                    lines!(ax_ratio, rr, fit.slope .* rr .* N_corr .+ fit.intercept;
                           scatter_kw(cD)...)
                end
            end
        end

        # ══════════════════════════════════════════════════════════════════
        # Row 2: ξ(D) vs D power-law fit with error band
        # ══════════════════════════════════════════════════════════════════

        v = xi_vs_D_corr[key]
            κ_corr, A_corr = v.κ_corr, v.A_corr
    
            # covariance from the fit (for the intercept and κ–lnA correlation) ...
            C = [v.κ_se^2      v.cov_κ_lnA
                 v.cov_κ_lnA   v.lnA_se^2]
            # ... but rescale the κ variance to the (wider) jackknife error, keeping the
            # κ–lnA correlation coefficient fixed so the band stays a valid covariance.
            κ_se_use = hasproperty(v, :κ_se_jack) && isfinite(v.κ_se_jack) ? v.κ_se_jack : v.κ_se
            ρ  = v.cov_κ_lnA / (v.κ_se * v.lnA_se)              # correlation coefficient
            C_band = [κ_se_use^2                ρ*κ_se_use*v.lnA_se
                      ρ*κ_se_use*v.lnA_se       v.lnA_se^2]
    
            A_se  = A_corr * v.lnA_se
            Dpr   = logrange(minimum(Dgood_corr), maximum(Dgood_corr), 100)
            logξ  = κ_corr .* log.(Dpr) .+ log(A_corr)
            selog = [ sqrt(max([log(D),1.0]' * C_band * [log(D),1.0], 0)) for D in Dpr ]
            ξ_hat, ξ_lo, ξ_hi = exp.(logξ), exp.(logξ .- selog), exp.(logξ .+ selog)
    
        ax_xi = Axis(grid[2, j],
            xlabel = "𝐷", ylabel = rich("𝜉",subscript("𝐷")),
         title=rich("𝜉",subscript("𝐷"), " = ($(round(A_corr; sigdigits=3)) ± $(round(A_se; sigdigits=2)))⋅𝐷",
                                superscript("($(round(κ_corr; sigdigits=3)) ± $(round(κ_se_use; sigdigits=2)))")),
                     titlefont = :regular,
                     titlesize = round(Int,figure_fontsize*0.8),
            xscale = log10, yscale = log10,
            xticks = (1:12, [string.(1:9)..., "", "", ""]),
            xminorticksvisible = false, xminorgridvisible = false)
        ξs_corr = fill(NaN, 12)
        if iN_corr !== nothing && haskey(xi_by_D, key)
            for D in 1:12
                fit = xi_by_D[key][iN_corr][D]
                (isfinite(fit.A) && isfinite(fit.ξ)) && (ξs_corr[D] = fit.ξ)
            end
        end
        isgood = map(dd -> in(dd, Dgood_corr), 1:12)
        scatter!(ax_xi, (1:12)[isgood],   ξs_corr[isgood];
                 markersize = 8, strokewidth = 1, color = Makie.wong_colors()[1])
        scatter!(ax_xi, (1:12)[.!isgood], ξs_corr[.!isgood];
                 markersize = 8, strokewidth = 0, color = Makie.wong_colors()[1])
        xlims!(ax_xi, 0.94, 13)

        # if length(Dgood_corr) ≥ 3 && haskey(xi_vs_D, key)
        #     v = xi_vs_D[key]
        #     κ_c, A_c = v.κ_corr, v.A_corr
        #     C = [v.κ_se^2      v.cov_κ_lnA
        #          v.cov_κ_lnA   v.lnA_se^2]           # 2×2 covariance of [κ, ln A]
        #     A_se  = A_c * v.lnA_se
        #     Dpr   = logrange(minimum(Dgood_corr), maximum(Dgood_corr), 100)
        #     logξ  = κ_c .* log.(Dpr) .+ log(A_c)
        #     selog = [ sqrt(max([log(D),1.0]' * C * [log(D),1.0], 0)) for D in Dpr ]
        #     band!(ax_xi, Dpr, exp.(logξ .- selog), exp.(logξ .+ selog);
        #           color = (:crimson, band_alpha))
        #     lines!(ax_xi, Dpr, exp.(logξ); color = :crimson,
        #            label = rich("($(round(A_c;sigdigits=3))±$(round(A_se;sigdigits=2)))⋅𝐷",
        #                         superscript("($(round(κ_c;sigdigits=3))±$(round(v.κ_se;sigdigits=2)))")))
        #     axislegend(ax_xi, position = :lt, padding = (2,2,-1,-1),
        #                framevisible = false, labelsize = 8)
        # end
        # κ_corr = A_corr = NaN
        if length(Dgood_corr) ≥ 3
    
            band!(ax_xi, Dpr, ξ_lo, ξ_hi; color = (:crimson, 0.22))
            lines!(ax_xi, Dpr, ξ_hat; color = :crimson,
                   label = "fit")
            axislegend(ax_xi, position = :lt, padding = (2,2,-1,-1), framevisible=false,)
        end

        # ══════════════════════════════════════════════════════════════════
        # Row 3: δE_abs vs L  
        # ══════════════════════════════════════════════════════════════════
        axE0 = Axis(grid[3, j],
            xlabel = "𝐿", ylabel = rich("𝛿𝐸", subscript("abs")),
            xscale = log2, yscale = log10,
            yticks = (10.0.^(-16:2:2), [rich("10", superscript("$(sign(x)==-1 ? "−" : "")$(abs(x))")) for x in -16:2:2]),
            xticks = (2.0.^(0:2:10), [rich("2", superscript("$(x)")) for x in 0:2:10]))
        for (cD, D) in enumerate(D_plot)
            iD = findfirst(==(D), D_scan); isnothing(iD) && continue
            sel_fss = FSS_filter[:, iD]; sel_co = .!sel_fss
            ys = results[key].δE0[:, iD]
            scatter!(axE0, N_scan[sel_co], ys[sel_co];
                     scatter_kw(cD)..., marker = markers[sel_co])
            scatter!(axE0, N_scan[sel_fss], ys[sel_fss];
                     scatter_kw(cD)..., marker = markers[sel_fss],strokewidth=0.8)
        end
        let v = filter(x -> isfinite(x) && x > 0, vec(results[key].δE0))
            isempty(v) || ylims!(axE0, 0.4*minimum(v), 5*maximum(v))
        end

        # ══════════════════════════════════════════════════════════════════
        # Row 4: FSS plateau | FDS plateau  (nested sub-grid)   (was row 2)
        # ══════════════════════════════════════════════════════════════════
        sub = grid[4, j] = GridLayout()

        # (left) finite-SIZE plateau: L²|δe_bulk| vs L/D^κ
        ax_FSS = Axis(sub[1, 1],
            xscale = log10, yscale = log10,
            xlabel = rich("𝐿/𝐷", superscript("$(κ_print)")),
            ylabel = rich("𝐿", superscript("2"), " |𝛿𝑒", subscript("bulk"),"|"),
            xticks = (
                10.0 .^ plateau_xtick_exponents,
                [rich("10", superscript("$(sign(x)==-1 ? "−" : "")$(abs(x))"))
                 for x in plateau_xtick_exponents],
            ),
            xticklabelsize = round(Int, 0.8*figure_fontsize), yticklabelsize = round(Int, 0.8*figure_fontsize),
            xminorticks = IntervalsBetween(9), yminorticks = IntervalsBetween(9))
        y_lo_cut, y_hi_cut = 1e-3, 1e1
        x_lo_I, x_hi_I = Inf, -Inf
        for (cD, D) in enumerate(D_plot)
            iD = findfirst(==(D), D_scan); isnothing(iD) && continue
            xs  = N_scan ./ ξ(D)
            ys2 = results[key].density_pert[:, iD]
            sel_fss = FSS_filter[:, iD]; sel_co = .!sel_fss
            scatter!(ax_FSS, xs[sel_co],  ys2[sel_co];  scatter_kw(cD)..., marker = markers[sel_co],  alpha = alpha_plot)
            scatter!(ax_FSS, xs[sel_fss], ys2[sel_fss]; scatter_kw(cD)..., marker = markers[sel_fss], alpha = alpha_plot, strokewidth=0.8)
            for (x, y) in zip(xs, ys2)
                (isfinite(x) && isfinite(y) && y_lo_cut ≤ y ≤ y_hi_cut) || continue
                x_lo_I = min(x_lo_I, x); x_hi_I = max(x_hi_I, x)
            end
        end
        hlines!(ax_FSS, [casimir_plateau]; color=color_ref, label = casimir_label)
        hlines!(ax_FSS, [casimir_plateau * (1 + fss_tol)]; color=:gray, linestyle=(:dash, :dense))
        ylims!(ax_FSS, y_lo_cut, y_hi_cut)
        x_lo_I < x_hi_I && xlims!(ax_FSS, 0.7*x_lo_I, 1.4*x_hi_I)

        # (right) finite-DEPTH plateau: ξ²|δe_bulk| / ln(ξ/a) vs D^κ/L
        ax_E_D = Axis(sub[1, 2],
            ylabel = rich("𝜉", superscript("2"), " |𝛿𝑒", subscript("bulk"), "|/ln(𝜉/𝑎)"),
            xlabel = rich("𝐷", superscript("$(κ_print)"), "/𝐿"),
            xscale = log10, yscale = log10,
            xticks = (
                10.0 .^ plateau_xtick_exponents,
                [rich("10", superscript("$(sign(x)==-1 ? "−" : "")$(abs(x))"))
                 for x in plateau_xtick_exponents],
            ),
            xticklabelsize = round(Int, 0.8*figure_fontsize), yticklabelsize = round(Int, 0.8*figure_fontsize),
            xminorticks = IntervalsBetween(9), yminorticks = IntervalsBetween(9))
        y_lo_cut_D, y_hi_cut_D = 1e-3, 1e1
        x_lo_D, x_hi_D = Inf, -Inf
        for (cD, D) in enumerate(D_plot)
            iD = findfirst(==(D), D_scan); isnothing(iD) && continue
            xs = ξ(D) ./ N_scan
            lf = logfac_xi(D);  lf > 0 || continue
            ys = results[key].density_pert[:, iD] ./ (N_scan .^ 2) .* ξ(D)^(2) / lf
            sel_fss = FSS_filter[:, iD]; sel_co = .!sel_fss
            scatter!(ax_E_D, xs[sel_co],  ys[sel_co];  scatter_kw(cD)..., marker = markers[sel_co],  alpha = alpha_plot)
            scatter!(ax_E_D, xs[sel_fss], ys[sel_fss]; scatter_kw(cD)..., marker = markers[sel_fss], alpha = alpha_plot, strokewidth = 0.8)
            for (x, y) in zip(xs, ys)
                (isfinite(x) && isfinite(y) && y_lo_cut_D ≤ y ≤ y_hi_cut_D) || continue
                x_lo_D = min(x_lo_D, x); x_hi_D = max(x_hi_D, x)
            end
        end
        ylims!(ax_E_D, y_lo_cut_D, y_hi_cut_D)
        if x_lo_D < x_hi_D
            xlims!(ax_E_D, 0.7*x_lo_D, 1.4*x_hi_D)
        end
        colsize!(sub, 1, Relative(0.5))
        colsize!(sub, 2, Relative(0.5))
        colgap!(sub, 10)

        # ══════════════════════════════════════════════════════════════════
        # Row 5: running-coupling collapse + correlator inset   (was row 3)
        # ══════════════════════════════════════════════════════════════════
        axE = Axis(grid[5, j],
            ylabel = rich("𝐿⋅𝛿𝐸", subscript("abs"), "/ln(𝜉/𝑎)"),
            xlabel = rich("𝐿/𝐷", superscript("$(κ_print)")),
            xscale = log10, yscale = log10,
            xticks = (10.0.^(-1:1:3), [rich("10", superscript("$(sign(x)==-1 ? "−" : "")$(abs(x))")) for x in -1:1:3]),
            yticks = (10.0.^(-14:2:4), [rich("10", superscript("$(sign(x)==-1 ? "−" : "")$(abs(x))")) for x in -14:2:4]),
            xminorticks = IntervalsBetween(9), yminorticks = IntervalsBetween(9))
        x_lo_E, x_hi_E = Inf, -Inf
        y_lo_E, y_hi_E = Inf, -Inf
        for (cD, D) in enumerate(D_plot)
            iD = findfirst(==(D), D_scan); isnothing(iD) && continue
            lf = logfac_xi(D);  lf > 0 || continue
            xs      = N_scan ./ ξ(D)
            ys      = results[key].pert_data[:, iD] ./ lf
            sel_fss = FSS_filter[:, iD]; sel_co = .!sel_fss
            scatter!(axE, xs[sel_co],  ys[sel_co];  scatter_kw(cD)..., marker = markers[sel_co],  alpha = alpha_plot)
            # scatter!(axE, xs[sel_fss], ys[sel_fss]; scatter_kw(cD)..., marker = markers[sel_fss], alpha = alpha_plot, strokewidth=0.8)
            # for (x, y) in zip(xs, ys)
            for (x, y) in zip(xs[sel_co], ys[sel_co])
                (isfinite(y) && y > 0) || continue
                x_lo_E = min(x_lo_E, x); x_hi_E = max(x_hi_E, x)
                y_lo_E = min(y_lo_E, y); y_hi_E = max(y_hi_E, y)
            end
        end
        if x_lo_E < x_hi_E
            let pr = logrange(x_lo_E, x_hi_E, 51)
                lines!(axE, pr, offset2 .* pr .^ 2; color = color_ref,
                       label = rich("∝ (𝐿/𝐷", superscript("$(κ_print)"), ")", superscript("2")))
            end
            xlims!(axE, min(0.8*x_lo_E, 1.0), 1.2*x_hi_E)
            ylims!(axE, 0.05*y_lo_E, 1.1*y_hi_E)
        end
        axislegend(axE, position=:lt, padding=(2,2,0,0))

        # (inset) correlator collapse  L^η |C_xx(r = distance·L)|
        axCxx = Axis(grid[5, j];
            halign = 0.975, valign = 0.22,
            width  = Relative(0.41), height = Relative(0.3),
            ylabel = rich("𝐿", superscript(η_label), " |𝐶", subscript("𝑋𝑋"),"|"),
            xlabel = rich("𝐿/𝐷", superscript("$(κ_print)")),
            xscale = log10, yscale = log10,
            xticks = (10.0.^(-4:1:4), [rich("10", superscript("$(sign(x)==-1 ? "−" : "")$(abs(x))")) for x in -4:1:4]),
            yticks = (10.0.^(-4:2:1), [rich("10", superscript("$(sign(x)==-1 ? "−" : "")$(abs(x))")) for x in -4:2:1]),
            inset_kw...)
        x_lo_C, x_hi_C = Inf, -Inf
        for (cD, D) in enumerate(D_plot)
            iD = findfirst(==(D), D_scan); isnothing(iD) && continue
            Ns_here   = N_scan[:]
            N_eff_vec = Ns_here ./ ξ(D)
            sel_fss = FSS_filter[:, iD]; sel_co = .!sel_fss
            Cxx_vec = [
                let idx_c = round(Int, N*distance)
                    (haskey(correlator_dict, (N, D)) &&
                     1 ≤ idx_c ≤ length(correlator_dict[(N, D)])) ?
                        correlator_dict[(N, D)][idx_c] : NaN
                end
                for N in Ns_here
            ]
            ys = Ns_here .^ η_corr .* abs.(Cxx_vec)
            scatter!(axCxx, N_eff_vec[sel_co],  ys[sel_co];  scatter_kw(cD)..., marker = markers[sel_co],  alpha = alpha_plot)
            scatter!(axCxx, N_eff_vec[sel_fss], ys[sel_fss]; scatter_kw(cD)..., marker = markers[sel_fss], alpha = alpha_plot, strokewidth=0.8)
            for (x, y) in zip(N_eff_vec, ys)
                (isfinite(x) && isfinite(y) && y > 0) || continue
                x_lo_C = min(x_lo_C, x)
                y ≥ y_cut && (x_hi_C = max(x_hi_C, x))
            end
        end
        ylims!(axCxx, y_cut, 10)
        if x_lo_C < x_hi_C
            xlims!(axCxx, 0.75*x_lo_C, 1.25*x_hi_C)
            text!(axCxx, x_lo_C, 2y_cut; align=(:left,:bottom),
                  text="𝑟 = $(round(distance,sigdigits=2)) 𝐿", fontsize=8)
        end
        translate!(axCxx.blockscene, 0, 0, 150)

        # ══════════════════════════════════════════════════════════════════
        # Row 6: eigenmode occupations + |δν| collapse inset   (was row 4)
        # ══════════════════════════════════════════════════════════════════
        ax_nk = Axis(grid[6, j],
            ylabel = "𝜈(𝑘)", xlabel = "mode 𝑘 / 𝑁",
            xticks = (0:N_mode÷2:2N_mode, string.(0:0.5:2)),
            ylabelpadding = 8)
        lines!(ax_nk, 1.0 .- occupation_gs(N_mode);
               color = color_ref, linewidth = 2, label = "target")
        let plotrange = vcat(LinRange(1, N_mode, N_mode), reverse(LinRange(N_mode+1, 2N_mode, N_mode)))
            scatter!(ax_nk, plotrange[mode_idx], 1.0 .- occupation_up(N_mode)[mode_idx];
                     color = color_init, label = "initial", markersize = 7, alpha = alpha_plot)
        end
        for (iD, D) in enumerate(D_show)
            haskey(results[key].n_k, (N_mode, D)) || continue
            scatter!(ax_nk, 1.0 .- results[key].n_k[N_mode, D][mode_idx];
                     color = iD, colormap = cmap_D, colorrange = (1, length(D_show)),
                     markersize = 6, alpha = alpha_plot)
        end
        xlims!(ax_nk, 0, 2N_mode + 1)

        ax_dnu = Axis(grid[6, j],
            xscale = log10, yscale = log10,
            yticks = (10.0.^(-12:2:2), [rich("10", superscript("$(sign(x)==-1 ? "−" : "")$(abs(x))")) for x in -12:2:2]),
            xticks = (
                10.0 .^ mode_xtick_exponents,
                [rich("10", superscript("$(sign(x)==-1 ? "−" : "")$(abs(x))"))
                 for x in mode_xtick_exponents],
            ),
            width = Relative(0.305), height = Relative(0.50),
            halign = 0.98, valign = 0.98, backgroundcolor = :white,
            ylabel = "|δν|", xlabel = rich("|𝑘 − 𝑁| 𝜉",subscript("𝐷"),"/𝐿"),
            xticklabelsize = round(Int, 0.6*figure_fontsize), yticklabelsize = round(Int, 0.6*figure_fontsize),
            xlabelpadding = 0, ylabelpadding = 0, xlabelsize = round(Int, 0.7*figure_fontsize), ylabelsize = round(Int, 0.7*figure_fontsize),
            xminorticksvisible = true, yminorticksvisible = true,
            xminorgridvisible = true, yminorgridvisible = true)
        for (iD, D) in enumerate(D_show)
            haskey(results[key].n_k, (N_mode, D)) || continue
            ξ_ent    = ξ(D)
            ν_sorted = results[key].n_k[N_mode, D][mode_idx]
            δν       = ν_sorted - occupation_gs(N_mode)
            dk       = abs.(collect(1:2N_mode) .- N_mode)
            scatter!(ax_dnu, (dk .+ 0.5) .* ξ_ent ./ N_mode, abs.(δν);
                     color = iD, colormap = cmap_D, colorrange = (1, length(D_show)),
                     markersize = 4, alpha = alpha_plot)
        end
        translate!(ax_dnu.blockscene, 0, 0, 150)

        if j == 1
            cmap = cgrad(cmap_D, length(D_show), categorical = true)
            elems = [MarkerElement(color = cmap[i], marker = :circle, markersize = 10)
                     for i in eachindex(D_show)]
            leg_modes = Legend(grid[6, j], elems, string.(D_show), "𝐿 = $(N_mode), 𝐷 :";
                framevisible = true, rowgap = 0, colgap = 2, nbanks = 3,
                titleposition = :top, titlefont = :regular, patchlabelgap = -4,
                padding = (4, 7, 0, 4), titlegap = 4,
                tellheight = false, tellwidth = false, halign = 0.05, valign = 0.1)
            translate!(leg_modes.blockscene, 0, 0, 160)
        end

        # ── panel labels (a)–(g) ──────────────────────────────────────────
        Label(grid[1,j,TopLeft()], "(a)"; font = :bold, fontsize = figure_fontsize, tellwidth = false, halign = :left, valign = :top)
        Label(grid[2,j,TopLeft()], "(b)"; font = :bold, fontsize = figure_fontsize, tellwidth = false, halign = :left, valign = :top)
        Label(grid[3,j,TopLeft()], "(c)"; font = :bold, fontsize = figure_fontsize, tellwidth = false, halign = :left, valign = :top)
        Label(sub[1,1,TopLeft()],  "(d)"; font = :bold, fontsize = figure_fontsize, tellwidth = false, halign = :left, valign = :top)
        Label(sub[1,2,TopLeft()],  "(e)"; font = :bold, fontsize = figure_fontsize, tellwidth = false, halign = :left, valign = :top)
        Label(grid[5,j,TopLeft()], "(f)"; font = :bold, fontsize = figure_fontsize, tellwidth = false, halign = :left, valign = :top)
        Label(grid[6,j,TopLeft()], "(g)"; font = :bold, fontsize = figure_fontsize, tellwidth = false, halign = :left, valign = :top)

        # ── column header ─────────────────────────────────────────────────
        Label(grid[0, j], VARIANTS[key].label;
              font = :bold, fontsize = figure_fontsize, tellwidth = false, tellheight=false)

    end #loop plot keys

    # Apply sizing only after every indexed row/column has been instantiated;
    # Makie does not allow sizing the negative-index colorbar row beforehand.
    rowsize!(grid, -1, Relative(top_fraction))
    rowsize!(grid,  0, Relative(header_fraction))
    for (row, fraction) in enumerate(data_fractions)
        rowsize!(grid, row, Relative(fraction))
    end
    for col in 1:ncol
        colsize!(grid, col, Relative(1 / ncol))
    end

    colgap!(grid, 8)
    rowgap!(grid, 8)
    return f
end


##%
# Eigenmode-occupation figure
# Source Pluto cell: 07bd8707-c529-4e94-859a-590eb45e2d6a
function eigenmode_plot(r::CircuitResults, κ::Real;
						# κ = κs_CO_v2[key],
						A = 1.0,
						N_mode = 64,
						n_k_init = occupation_up,
						alpha_plot = 0.8,
	)
	# plot_key = key
	# r   = results[plot_key]
	# var = getfield(VARIANTS, plot_key)

	iN = findfirst( nn -> nn==N_mode, N_scan)
	
	k_plot = LinRange(0,2N_mode,501)
    
    # Mode index: 1 = lowest energy, 2N = highest energy
    mode_idx = vcat(1:N_mode,2N_mode:-1:N_mode+1)
    ε_sorted = map( k -> dispersion_relation(k,N_mode), k_plot)
	
	D_show = [1,2,3,6,12]


	ξ(D) = A * D^κ
	kappa = κ

		
    f_modes = Figure(size = (320, 200), figure_padding=(0,4,2,1))
    g_modes = f_modes[1, 1] = GridLayout()
    
	# Label(g_modes[0, 1], var.label; font = :bold, tellwidth = false)

	l_padd = (0,0,0,-4)

	let cmap = cgrad(cmap_D, length(D_show), categorical = true)
    elems = [MarkerElement(color = cmap[i], marker = :circle, markersize = 10)
             for i in eachindex(D_show)]
    Legend(g_modes[1, 1], (elems), (string.(D_show)), "𝐿 = $(N_mode), 𝐷 :";
		   # orientation = :horizontal,
		   framevisible = true,
		   rowgap = 0,
		   colgap = 2,
		   nbanks = 3,
		   titleposition = :top,
		   titlefont=:regular,
		   patchlabelgap = -4,
		   padding = (4, 7, 0,4),
		   titlegap = 4,
		   # labelsize=11,
		   tellheight = false,
		   tellwidth = false,
		   halign=0.05, valign=0.1,
		  )
	end
	# Label(g_modes[0, 1, Left()], "(a)"; font = :bold, tellwidth = false)
        
        
	# ── Row 1: mode occupatioN_scan vs mode index (sorted by energy) ─
	# Label(g_modes[1, 1, Left()], "(a)"; font = :bold, valign=:top, halign=:left, padding = (0,0,0,0))

	ax1 = Axis(g_modes[1, 1],
		ylabel = "ν(𝑘)",
		xlabel = "mode 𝑘 / 𝑁",
		xticks = (0:N_mode÷2:2N_mode, string.(0:0.5:2)),
		ylabelpadding = 8,
	)
	
	lines!(ax1,
		   # [0,N_mode-eps(),N_mode+eps(), 2N_mode], [1.0, 1.0, 0.0, 0.0];
		   1.0 .- occupation_gs(N_mode),
		   color = color_ref, linewidth = 2, linestyle = (:solid, :dense), label = "target")
	
	N_dense = N_mode
	
	mode_dense_idx = vcat(1:N_dense,2N_dense:-1:N_dense+1)
	plotrange = vcat(
		LinRange(1,N_mode,N_dense),
		reverse(LinRange(N_mode+1,2N_mode,N_dense))
	)
	scatter!(ax1, plotrange[mode_dense_idx], 1.0 .- n_k_init(N_dense)[mode_dense_idx];
			color = color_init,
			 # linewidth = 1, linestyle = (:solid, :dense),
			label = "initial",
			markersize=7,
			 alpha = alpha_plot,
		)

	
		
	
	for (iD,D) in enumerate(D_show)
		 
		if haskey(r.n_k, (N_mode,D))
		scatter!(ax1, 1.0 .- r.n_k[N_mode,D][mode_idx];
			color = iD, colormap = cmap_D,
			colorrange = (1, length(D_show)),
			# linewidth = 1,
				 markersize=6,
				 alpha = alpha_plot,
		)

		end
	end
	# axislegend(ax1, "𝑁 = $(N_mode)" ; titlefont=:regular, position = :lb)
	
	# # ── Row 3: |δν| vs |mode - N| · ξ_ent (collapse) ────────────
	# Label(g_modes[2, 1, Left()], "(c)"; font = :bold, valign=:top, halign=:left,padding = l_padd)
	ax3 = Axis(g_modes[1,1],
		ylabel = "|δν|",
		xlabel = rich("|𝑘 − 𝑁| 𝜉",subscript("𝐷"),"/𝐿"),
		yscale = log10, xscale = log10,
		width=Relative(0.33),
		height=Relative(0.55),
		halign=0.99,
		valign=0.95,
		xticklabelsize = 8,
		yticklabelsize = 8,
		ylabelpadding = 2,
		xlabelsize = 9,
		ylabelsize = 9,
		xminorticksvisible = false,
		yminorticksvisible = false,
		xminorgridvisible = false,
		yminorgridvisible = false, 
	)
	for (iD,D) in enumerate(D_show)

		if haskey(r.n_k, (N_mode,D))
		iDD = findfirst(x -> x == D, D_scan)
		ξ_ent = ξ(D)
	   
		ν_circuit_sorted = r.n_k[N_mode,D][mode_idx]
		δν = ν_circuit_sorted - occupation_gs(N_mode)
		dk = abs.(collect(1:2N_mode) .- N_mode)
		scatter!(ax3, (dk .+ 0.5) .* ξ_ent ./ N_mode, abs.(δν);
				 color = iD, colormap = cmap_D,
				 colorrange = (1, length(D_show)), markersize = 4,
				 alpha = alpha_plot,
		)
		end
	end
	# vlines!(ax3, [1.0]; color = :gray, linestyle = :dash)


	translate!(ax3.blockscene, 0, 0, 150)

	xlims!(ax1,0,2N_mode+1)
	# rowgap!(g_modes,1,5)
	# rowsize!(g_modes, 1, Relative(0.4))
	# rowsize!(g_modes, 2, Relative(0.3))
	# rowsize!(g_modes, 3, Relative(0.3))
	
	return f_modes
end


##%
# Network-parameter figure
# Source Pluto cell: c363ed64-c34b-41c4-8c31-70c6f95675c6
"""
    network_parameter_plot(; display_keys=loop_keys, D_values=1:12,
                             N_show=128, savepath=nothing) -> Figure

Plot the absolute layer parameters for every retained Ising ansatz. When
`savepath` is provided, save the figure there.
"""
function network_parameter_plot(; display_keys = loop_keys,
                                  D_values = 1:12,
                                  N_show = 128,
                                  savepath = nothing)
    nrows   = length(display_keys)
    ncols   = 3   # maximum parameters per layer across all variants

    # f = Figure(size = (ncols * 260, nrows * 160))
    f = Figure(
        size = (680, 870),
        fontsize = 9,
        figure_padding = (5, 8, 5, 5),
    )
    g = f[1, 1] = GridLayout()

    Colorbar(
        g[1:2, 4],
        limits   = (0.5, length(D_values) + 0.5),
        colormap = cgrad(cmap_D, length(D_values), categorical=true),
        ticks    = (1:length(D_values), string.(D_values)),
        label    = "circuit depth 𝐷",
		# vertical= false,
		tellwidth=false,
    )

    line_kw(iD) = (
        color      = iD,
        colormap   = cmap_D,
        colorrange = (1, length(D_values)),
    )

    for (row, key) in enumerate(display_keys)
        r   = results[key]
        var = getfield(VARIANTS, key)
        nP  = var.n_params

        Label(g[row, 1], var.label; font = :bold, tellheight = false, rotation = π/2)

        axes = [Axis(g[row, col + 1],
                    ylabel = var.params_labels[col],
                    xlabel = row == nrows ? "layer 𝑑" : "",
                    xticks = 0:2:22,
                    xminorticksvisible = true,
					 xminorticks = IntervalsBetween(2),
                ) for col in 1:nP]

        # blank out unused columns so the grid stays uniform
        for col in nP+1:ncols
            ax = Axis(g[row, col + 1])
            hidedecorations!(ax)
            hidespines!(ax)
        end

        for (iD, D) in enumerate(D_values)
            haskey(r.params, (N_show, D)) || continue
            p_vec = r.params[(N_show, D)]
            length(p_vec) == D * nP || continue
            for col in 1:nP
                vals = abs.(p_vec[col:nP:end])
                scatterlines!(axes[col], 1:D, vals; line_kw(iD)..., linestyle=(:dot, :dense))
            end
        end
    end

    colgap!(g, 8)
    rowgap!(g, 6)

    isnothing(savepath) || save(savepath, f)
    return f
end

##%
# Example publication figures
const EXAMPLE_ANSATZ = :Rxx_Rz
const NN_ANSATZE = [:Rxx_Rz, :TFI_only, :kitaev_HVA, :kitaev_only]
const LONG_RANGE_ANSATZE = [:EXP_HVA, :TFI_EXP, :POW_HVA, :TFI_POW]

mkpath(plotdir)

##%
# Correlator and correlation-length scaling for the example ansatz
example_correlator_figure = plot_xi_corr_panel(EXAMPLE_ANSATZ;
    xi_by_D = xi_corr_by_D,
)
save(
    joinpath(plotdir, "fig-main-Corr_scaling_$(EXAMPLE_ANSATZ)_vert.pdf"),
    example_correlator_figure,
)
example_correlator_figure

##%
# Running-coupling collapse for the example ansatz
example_running_coupling_figure = let
    key = EXAMPLE_ANSATZ
    classify_settings = CLASSIFY_SETTINGS[key]
    plot_settings = PLOT_SETTINGS[key]

    make_figure_vert_Ising(results[key];
        κ = κs_v4[key].κ,
        A_xi = xi_vs_D_corr[key].A_corr,
        a_uv = κs_v4[key].a,
        FSS_filter = results[key].FSS_filter_v2,
        D_plot = plot_settings.D_plot,
        D_fit = classify_settings.D_fit,
        N_cap_FDS = plot_settings.N_cap_FDS,
        casimir_label_FDS = classify_settings.casimir_label_FDS,
    )
end
save(
    joinpath(plotdir, "fig-main-collapse_scaling_$(EXAMPLE_ANSATZ)_vert.pdf"),
    example_running_coupling_figure,
)
example_running_coupling_figure

##%
# Eigenmode occupations for the example ansatz
example_eigenmode_figure = eigenmode_plot(
    results[EXAMPLE_ANSATZ],
    xi_vs_D_corr[EXAMPLE_ANSATZ].κ_corr;
    A = xi_vs_D_corr[EXAMPLE_ANSATZ].A_corr,
    N_mode = N_mode_plot,
)
save(
    joinpath(plotdir, "fig-EigenModes_$(EXAMPLE_ANSATZ)_N$(N_mode_plot).pdf"),
    example_eigenmode_figure,
)
example_eigenmode_figure

##%
# Four nearest-neighbour ansätze, sized as one A4-friendly megaplot
mega_plot_nn = make_mega_plot(NN_ANSATZE;
    kappa_dict = κs_v4,
    xi_by_D = xi_corr_by_D,
)
save(joinpath(plotdir, "fig-mega-plot-nearest-neighbour.pdf"), mega_plot_nn)
mega_plot_nn

##%
# Four long-range ansätze, sized as a second A4-friendly megaplot
mega_plot_long_range = make_mega_plot(LONG_RANGE_ANSATZE;
    kappa_dict = κs_v4,
    xi_by_D = xi_corr_by_D,
)
save(joinpath(plotdir, "fig-mega-plot-long-range.pdf"), mega_plot_long_range)
mega_plot_long_range

##%
# Optimized network parameters for all eight ansätze
let N_show = 128
    network_parameters_figure = network_parameter_plot(
    display_keys = [NN_ANSATZE; LONG_RANGE_ANSATZE],
    D_values = 1:12,
    N_show = N_show,
    savepath = joinpath(plotdir, "fig-network-parameters-N$(N_show).pdf"),
)
network_parameters_figure
end

##%
# Script entry point
if abspath(PROGRAM_FILE) == @__FILE__
    display(central_findings_table())
end

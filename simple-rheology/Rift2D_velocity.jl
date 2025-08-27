const isCUDA = false
#const isCUDA = true

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
import JustRelax.@cell

const backend_JR = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustRelax.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using ParallelStencil, ParallelStencil.FiniteDifferences2D

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using JustPIC, JustPIC._2D
# Threads is the default backend_JR,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend_JP = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

# Load script dependencies
using GeoParams, CairoMakie
#using  CairoMakie
using JLD2


# Load file with all the rheology configurations
include("Rift2DSetupFaultInclusion.jl")
include("RiftRheology.jl")


## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

import ParallelStencil.INDICES
const idx_k = INDICES[2]
macro all_k(A)
    esc(:($A[$idx_k]))
end

function copyinn_x!(A, B)
    @parallel function f_x(A, B)
        @all(A) = @inn_x(B)
        return nothing
    end

    @parallel f_x(A, B)
end

@parallel_indices (i, j) function update_Dirichlet_mask!(mask, phase_ratio_vertex, air_phase)
    @inbounds mask[i + 1, j] = @index(phase_ratio_vertex[air_phase, i, j]) == 1
    nothing
end

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_k(z)) #* <(@all_k(z), 0.0)
    return nothing
end

function apply_pure_shear(Vx, Vy, εbg, xvi, lx, ly)
    xv, yv = xvi

    @parallel_indices (i, j) function pure_shear_x!(Vx, εbg, lx)
        xi = xv[i]
        Vx[i, j + 1] = εbg * (xi - lx * 0.5)
        return nothing
    end

    @parallel_indices (i, j) function pure_shear_y!(Vy, εbg, ly)
        yi = yv[j]
        Vy[i + 1, j] = abs(yi) * εbg
        return nothing
    end

    nx, ny = size(Vx)
    @parallel (1:nx, 1:(ny - 2)) pure_shear_x!(Vx, εbg, lx)
    nx, ny = size(Vy)
    @parallel (1:(nx - 2), 1:ny) pure_shear_y!(Vy, εbg, ly)

    return nothing
end

function apply_pure_shear(Ux, Uy, εbg, xvi, lx, ly, dt)
    xv, yv = xvi

    @parallel_indices (i, j) function pure_shear_x!(Ux, εbg, lx, dt)
        xi = xv[i]
        Ux[i, j + 1] = εbg * (xi - lx * 0.5) * dt
        return nothing
    end

    @parallel_indices (i, j) function pure_shear_y!(Uy, εbg, ly, dt)
        yi = yv[j]
        Uy[i + 1, j] = abs(yi) * εbg *dt
        return nothing
    end

    nx, ny = size(Ux)
    @parallel (1:nx, 1:(ny - 2)) pure_shear_x!(Ux, εbg, lx, dt)
    nx, ny = size(Uy)
    @parallel (1:(nx - 2), 1:ny) pure_shear_y!(Uy, εbg, ly, dt)

    return nothing
end

## END OF HELPER FUNCTION ------------------------------------------------------------

## END OF MAIN SCRIPT ----------------------------------------------------------------
do_vtk   = true # set to true to generate VTK files for ParaView
figdir   = "output/Rift2D_displacement"
n        = 128
nx, ny   = n, n ÷ 2
# li, origin, phases_GMG, T_GMG = Setup_Topo(nx+1, ny+1)
li, origin, phases_GMG, T_GMG = flat_setup(nx+1, ny+1)
# heatmap(T_GMG)
# heatmap(phases_GMG)

nx, ny = size(T_GMG).-1

igg      = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI= true)...)
else
    igg
end

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
#function main(li, origin, phases_GMG, igg; nx=16, ny=16, figdir="figs2D", do_vtk =false)

    # Physical domain ------------------------------------
    ni                  = nx, ny           # number of cells
    di                  = @. li / ni       # grid steps
    grid                = Geometry(ni, li; origin = origin)
    (; xci, xvi)        = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology     = init_rheologies(; incompressible=false, linear = true)
    #rheology_incomp = init_rheology(; is_compressible=false, linear = false)
    dt_time = 1.0e3 * 3600 * 24 * 365
    k = (4 / (rheology[1].HeatCapacity[1].Cp.val * rheology[1].Density[1].ρ0.val)) # thermal diffusivity   # thermal diffusivity
    dt_diff = 0.5 * min(di...)^2 / k / 2.01
    dt = min(dt_time, dt_diff)
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell    = 75
    max_xcell = 100
    min_xcell = 50
    particles = init_particles(
        backend_JP, nxcell, max_xcell, min_xcell, xvi, di, ni
    )
    # velocity grids
    grid_vxi = velocity_grids(xci, xvi, di)
    # material phase & temperature
    subgrid_arrays = SubgridDiffusionCellArrays(particles)
    pPhases, pT    = init_cell_arrays(particles, Val(2))
    particle_args  = (pT, pPhases)
    # Assign particles phases anomaly
    phases_device = PTArray(backend_JR)(phases_GMG)
    phase_ratios  = phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phases!(pPhases, phases_device, particles, xvi)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
    # ----------------------------------------------------
 
    # RockRatios
    air_phase = 4
    ϕ_R = RockRatio(backend_JR, ni)
    update_rock_ratio!(ϕ_R, phase_ratios, air_phase)
 
    # marker chain
    nxcell, min_xcell, max_xcell = 100, 75, 125
    initial_elevation = 0e3
    chain = init_markerchain(backend_JP, nxcell, min_xcell, max_xcell, xvi[1], initial_elevation)
    update_phases_given_markerchain!(pPhases, chain, particles, origin, di, air_phase)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
    update_rock_ratio!(ϕ_R, phase_ratios, air_phase)

    # particle fields for the stress rotation
    pτ = StressParticles(particles)
    particle_args = (pT, pPhases, unwrap(pτ)...)
    particle_args_reduced = (pT, unwrap(pτ)...)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes           = StokesArrays(backend_JR, ni)
    pt_stokes        = PTStokesCoeffs(li, di;  ϵ_rel=1e-5,ϵ_abs=1e-4, Re=3*√10*π/2, r=0.5, CFL = 0.85 / √2.1) # Re=3π, r=0.7
    σ = PrincipalStress(backend_JR, ni)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    Ttop             = minimum(T_GMG)
    Tbot             = maximum(T_GMG)
    thermal          = ThermalArrays(backend_JR, ni)
    @views thermal.T[2:end-1, :] .= PTArray(backend_JR)(T_GMG)

    # Add thermal anomaly BC's
    T_air = 273.0e0
    Ω_T = @zeros(size(thermal.T)...)
    
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux     = (; left = true, right = true, top = false, bot = false),
        dirichlet   = (; constant = T_air, mask = Ω_T)
    )
    thermal_bcs!(thermal, thermal_bc)
    @views thermal.T[:, end] .= Ttop
    @views thermal.T[:, 1]   .= Tbot
    temperature2center!(thermal)
    # ----------------------------------------------------

    # Buoyancy forces
    ρg               = ntuple(_ -> @zeros(ni...), Val(2))
    compute_ρg!(ρg[2], phase_ratios, rheology, (T=thermal.Tc, P=stokes.P))
    stokes.P        .= PTArray(backend_JR)(reverse(cumsum(reverse((ρg[2]).* di[2], dims=2), dims=2), dims=2))

    # Rheology
    args0            = (T=thermal.Tc, P=stokes.P, dt = Inf)
    viscosity_cutoff = (1e17, 1e25)
    compute_viscosity!(stokes, phase_ratios, args0, rheology, viscosity_cutoff; air_phase = air_phase)

    # PT coefficients for thermal diffusion
    pt_thermal       = PTThermalCoeffs(
        backend_JR, rheology, phase_ratios, args0, dt, ni, di, li; ϵ=1e-5, CFL=0.98 / √2.1
    )


    flow_bcs         = VelocityBoundaryConditions(;
        free_slip    = (left = true , right = true , top = true , bot = true),
        free_surface = false,
    )

    εbg = +8.64e-16 # background strain rate
    stokes.V.Vx[:, 2:(end - 1)] .= PTArray(backend_JR)([ εbg * x  for x in xvi[1], y in xci[2]])
    stokes.V.Vy[2:(end - 1), :] .= PTArray(backend_JR)([-εbg * y  for x in xci[1], y in xvi[2]])

    # BC_topography_displ(stokes.U.Ux, stokes.U.Uy, εbg, xvi, li..., dt)
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    #displacement2velocity!(stokes, dt)
    update_halo!(@velocity(stokes)...)

    # IO -------------------------------------------------
    # if it does not exist, make folder where figures are stored
    if do_vtk
        vtk_dir      = joinpath(figdir, "vtk")
        fig_dir      = joinpath(figdir, "fig")
        take(vtk_dir)
        take(fig_dir)
    end
    take(figdir)
    
    # ----------------------------------------------------

    local Vx_v, Vy_v,Ux_v, Uy_v
    if do_vtk
        Vx_v = @zeros(ni.+1...)
        Vy_v = @zeros(ni.+1...)
        Ux_v = @zeros(ni.+1...)
        Uy_v = @zeros(ni.+1...)
    end

    T_buffer    = @zeros(ni.+1)
    Told_buffer = similar(T_buffer)
    dt₀         = similar(stokes.P)
    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end
    grid2particle!(pT, xvi, T_buffer, particles)

    # Time loop
    t, it = 0.0, 0

    # uncomment for random cohesion damage
    # cohesion_damage = @rand(ni...) .* 0.05 # 5% random cohesion damage

    while it < 10 # run only for 5 Myrs

        # BC_topography_displ(stokes.U.Ux, stokes.U.Uy, εbg, xvi, li..., dt)
        # flow_bcs!(stokes, flow_bcs) # apply boundary conditions
        # update_halo!(@velocity(stokes)...)

        # interpolate fields from particle to grid vertices
        particle2grid!(T_buffer, pT, xvi, particles)
        @views T_buffer[:, end]      .= Ttop
        @views T_buffer[:, 1]        .= Tbot
        @views thermal.T[2:end-1, :] .= T_buffer
        thermal_bcs!(thermal, thermal_bc)
        temperature2center!(thermal)

        # args = (; T = thermal.Tc, P = stokes.P,  dt=Inf, cohesion_C = cohesion_damage)
        args = (; T = thermal.Tc, P = stokes.P,  dt=Inf)

        # Stokes solver ----------------
        result, t_stokes,_,_,_ = @timed begin
            solve_VariationalStokes!(
                stokes,
                pt_stokes,
                di,
                flow_bcs,
                ρg,
                phase_ratios,
                ϕ_R,
                rheology,
                args,
                dt,
                igg;
                kwargs = (;
                    iterMax = it > 0 ? 250.0e3 : 500.0e4,
                    free_surface = true,
                    strain_increment=true,
                    nout = 1e3,
                    viscosity_cutoff = viscosity_cutoff,
                )
            );
        end;
        # dtmax = (it < 10 ? 2e3 : 3e3) * 3600 * 24 * 365 # diffusive CFL timestep limiter
        #dt    = compute_dt(stokes, di, dtmax)
        println("Stokes solver time             ")
        println("   Total time:      $t_stokes s")
        println("           Δt:      $(dt / (3600 * 24 * 365)) kyrs")
        # println("   Time/iteration:  $(t_stokes / out.iter) s")
        # ------------------------------

        compute_shear_heating!(
            thermal,
            stokes,
            phase_ratios,
            rheology, # needs to be a tuple
            dt,
        )

        # Thermal solver ---------------
        # update mask of Dirichlet BC
        @parallel (@idx ni .+ 1) update_Dirichlet_mask!(thermal_bc.dirichlet.mask, phase_ratios.vertex, air_phase)
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            dt,
            di;
            kwargs = (
                igg     = nothing,
                phase   = phase_ratios,
                iterMax = 50e4,
                nout    = 1e3,
                verbose = true,
            )
        )
        @show extrema(thermal.T)

        subgrid_characteristic_time!(
            subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes, xci, di
        )
        centroid2particle!(subgrid_arrays.dt₀, xci, dt₀, particles)
        subgrid_diffusion!(
            pT, thermal.T, thermal.ΔT, subgrid_arrays, particles, xvi,  di, dt
        )
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection_MQS!(particles, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (pT, ), (T_buffer, ), xvi)

        # advect marker chain
        advect_markerchain!(chain, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
        update_phases_given_markerchain!(pPhases, chain, particles, origin, di, air_phase)

        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
        update_rock_ratio!(ϕ_R, phase_ratios, air_phase)
  

        @show it += 1
        t        += dt

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 1) == 0
            # checkpointing(figdir, stokes, thermal.T, η, t)
            (; η_vep, η) = stokes.viscosity
            if do_vtk

                velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
                velocity2vertex!(Ux_v, Uy_v, @displacement(stokes)...)
                data_v = (;
                    T = Array(T_buffer),
                    Vx = Array(Vx_v),
                    Vy = Array(Vy_v),
                    Ux = Array(Ux_v),
                    Uy = Array(Uy_v),
                )
                data_c = (;
                    Pdyno = Array(stokes.P) .- Array(reverse(cumsum(reverse((ρg[2]) .* di[2], dims = 2), dims = 2), dims = 2)),
                    P = Array(stokes.P),
                    η = Array(η),
                    η_vep = Array(η_vep),
                    τII = Array(stokes.τ.II),
                    τxx = Array(stokes.τ.xx),
                    τyy = Array(stokes.τ.yy),
                    εII = Array(stokes.ε.II),
                    εII_pl = Array(stokes.ε_pl.II),
                    ρ = Array( ρg[2]/9.81),
                )
                velocity_v = (
                    Array(Vx_v),
                    Array(Vy_v),
                )
                save_marker_chain(
                    joinpath(vtk_dir, "topo_" * lpad("$it", 6, "0")),
                    xvi[1],
                    Array(chain.h_vertices),
                )
                save_vtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                    xvi,
                    xci,
                    data_v,
                    data_c,
                    velocity_v
                )
            end

            checkpointing_jld2(
                figdir,
                stokes,
                thermal,
                t,
                dt,
            )

            checkpointing_particles(
                figdir,
                particles,
                phases = pPhases,
                phase_ratios = phase_ratios,
                chain = chain,
                t = t,
                dt = dt,
                particle_args = particle_args,
            )


            fig = Figure(resolution = (800, 400))

            # Add an axis
            ax = Axis(fig[1, 1],  title = "Err vs PT-Iteration", xlabel = "Pseudo-Iteration", ylabel = "Error Abs")
            lines!(ax, 1:length(result.err_evo1), result.err_evo1, color = :blue)
            scatter!(ax, 1:length(result.err_evo1), result.err_evo1, color = :red)
            savefig(p, 
            joinpath(fig_dir, "err_" * lpad("$it", 6, "0")))

            # Make particles plottable
            tensor_invariant!(stokes.ε)
            tensor_invariant!(stokes.ε_pl)

            # allocate CellArrays to store the velocity field of the marker chain
            chain_V = similar(chain.coords[1]), similar(chain.coords[1])
            chain_V[1].data .*= 0
            chain_V[2].data .*= 0

            interpolate_velocity_to_markerchain!(chain, chain_V, (stokes.V.Vx, stokes.V.Vy), grid_vxi)
            args = Dict(
                :Vx_chain => Array(chain_V[1]),
                :Vy_chain => Array(chain_V[2]),
                :x => xvi[1],
                :y => Array(chain.h_vertices)
            )
            jldsave(joinpath(vtk_dir, "chain_" * lpad("$it", 6, "0") * ".jld2"); args...)

        end
        # ------------------------------

    end



    return nothing
#end




#main(li, origin, phases_GMG, igg; figdir = figdir, nx = nx, ny = ny, do_vtk = do_vtk);


using Distributed
using LinearAlgebra:I,diagm,diag
using TensorKit
using Random
using Printf
using DelimitedFiles
using JLD2


cd(@__DIR__)




# include("../../../../state/iPEPS_ansatz.jl")

# include("../../../../setting/Settings.jl")
# include("../../../../setting/linearalgebra.jl")
# include("../../../../setting/tuple_methods.jl")

include("MC/contract_torus.jl")
include("MC/sampling.jl")
include("MC/sampling_eliminate_physical_leg.jl")
include("MC/build_degenerate_states.jl")

begin
@show const Lx = 6      # number of sites along x / number of columns in the lattice
@show const Ly = 6      # number of sites along y / number of rows in the lattice
@show const D=3;#bond dimension of state
@show const chi=10;#bond dimension of environment

const L = Lx * Ly # total number of lattice sites
const Nbra = L             # Inner loop size, to generate uncorrelated samples, usually must be of size O(L).
@show const Nsteps = 400000       # Total Monte Carlo steps
@show const binn = 1000          # Bin size to store the data during the monte carlo run. 
const GC_spacing = 200          # garbage collection
end

###################

###################




####################
#use single core
import LinearAlgebra.BLAS as BLAS
using Base.Threads

n_cpu=1;
BLAS.set_num_threads(n_cpu);
println("number of cpus: "*string(BLAS.get_num_threads()));flush(stdout);

Base.Sys.set_process_title("C"*string(n_cpu)*"_fPEPS")
pid=getpid();
println("pid="*string(pid));;flush(stdout);
####################

#input  TensorKit.   and then press Tab to show all properties that may occupy memory

# TensorKit.usebraidcache_abelian[] = false 
# TensorKit.usebraidcache_nonabelian[] = false
TensorKit.braidcache.maxsize=1000
TensorKit.transposecache.maxsize=1000
# TensorKit.usetransposecache
TensorKit.treepermutercache.maxsize=1000
TensorKit.GLOBAL_FUSIONBLOCKSTRUCTURE_CACHE.maxsize=1000
# Base.summarysize(TensorKit.treepermutercache)
# Base.summarysize(TensorKit.GLOBAL_FUSIONBLOCKSTRUCTURE_CACHE)



function overlap_ratio(iconf_new::Vector,sample_::Matrix{TensorMap}, amplitude::Number, contract_history_otherstate_::Contract_History)
    # Compute the expectation value of the permutation operator
    
    global contract_fun,psi_decomposed_otherstate,Vp

    if contraction_path=="verify"
        amplitude_otherstate,sample_,_=contract_sample(psi_decomposed_otherstate,Lx,Ly,iconf_new,sample_, Vp,contract_fun);
        amplitude_otherstate_,sample_,_,contract_history_= partial_contract_sample(psi_decomposed_otherstate,iconf_new,sample_, Vp,contract_history_otherstate_);
        @assert abs(norm(amplitude_otherstate-amplitude_otherstate_)/amplitude_otherstate)<1e-10;
    elseif contraction_path=="full"
        amplitude_otherstate,sample_,_=contract_sample(psi_decomposed_otherstate,Lx,Ly,iconf_new,sample_, Vp,contract_fun);
    elseif contraction_path=="recycle"
        amplitude_otherstate,sample_,_,contract_history_otherstate_= partial_contract_sample(psi_decomposed_otherstate,iconf_new,sample_, Vp,contract_history_otherstate_);
    end


    elocal = amplitude_otherstate/amplitude;

        
    return elocal,sample_, contract_history_otherstate_
end



function main(dir::String, worker_id::Int, ntask::Int, BC1::Int,BC2::Int)
    #load saved fPEPS data
    @show Nsteps_worker=Int(round(Nsteps/ntask));
    contraction_path="recycle";#"verify","full","recycle"
 
    filenm="CSL_D"*string(D)*"_U1";
    @show to_dense=false;#convert to dense
    psi0,Vp,Vv=load_fPEPS_from_iPEPS(Lx,Ly,filenm,to_dense);

    global contraction_path, contract_fun, psi_decomposed,psi_decomposed_otherstate, Vp, projector_method
    projector_method="1";#"1" or "2"
    contract_fun=contract_whole_torus_boundaryMPS;
    normalize_PEPS!(psi0,Vp,contract_whole_torus_boundaryMPS);#normalize psi0 such that the amplitude of a single config is close to 1
    #psi_00,psi_0pi,psi_pi0,psi_pipi =construct_4_states(psi0,Vv);#four states
    psi_BC_set =construct_4_states(psi0,Vv);#four states

    psi=psi_BC_set[BC1];
    psi_decomposed=decompose_physical_legs(psi,Vp);

    psi_otherstate=psi_BC_set[BC2];
    psi_decomposed_otherstate=decompose_physical_legs(psi_otherstate,Vp);
    sample0=Matrix{TensorMap}(undef,Lx,Ly);
    sample1=Matrix{TensorMap}(undef,Lx,Ly);
    
    ##########################################

    coord,fnn_set,snn_set,NN_tuple,NNN_tuple, NN_tuple_reduced,NNN_tuple_reduced=get_neighbours(Lx,Ly,"PBC");

    initial_iconf =initial_Neel_config(Lx,Ly,1);
    #Recall that iconf here has elements 1 (up spin) and -1 (down spin), unlike in our C++ code where we have 1 and 2.

    #start from the config in test.csv


    #create empty contract_history
    contract_history=torus_contract_history(zeros(Int8,Lx*Ly),Matrix{TensorMap}(undef,Lx,Ly));
    contract_history_otherstate=torus_contract_history(zeros(Int8,Lx*Ly),Matrix{TensorMap}(undef,Lx,Ly));


    # Initialize variables
    iconf_new = copy(initial_iconf)

    ebin1 = zeros(Complex{Float64}, binn)
    
    outputname = dir*"id_"*string(worker_id)*"_BC"*string(BC1)*"_"*string(BC2)*"_chi"*string(chi)*".csv"

    if isfile(outputname)
        rm(outputname)
    end

    open(outputname, "a") do file # "a" is for append

        # @inbounds for i in 1:Nsteps_worker  # Number of Monte Carlo steps, usually 1 million
        #     @inbounds for j in 1:Nbra  # Inner loop to create uncorrelated samples
        for i in 1:Nsteps_worker  # Number of Monte Carlo steps, usually 1 million
            global ite_num
            ite_num=i;
            # @show i
            # if mod(i,100)==0;@show i;flush(stdout);end


            if contraction_path=="verify"
                amplitude,sample0, _=contract_sample(psi_decomposed,Lx,Ly,iconf_new,sample0,Vp,contract_fun);
                amplitude_, sample0, _,contract_history= partial_contract_sample(psi_decomposed,iconf_new,sample0,Vp,contract_history);
                @assert abs(norm(amplitude-amplitude_)/amplitude)<1e-10;

                amplitude_otherstate,sample1,_=contract_sample(psi_decomposed_otherstate,Lx,Ly,iconf_new,sample1,Vp,contract_fun);
                amplitude_otherstate_,sample1,_,contract_history= partial_contract_sample(psi_decomposed_otherstate,iconf_new,sample1,Vp,contract_history_otherstate);
                @assert abs(norm(amplitude_otherstate-amplitude_otherstate_)/amplitude_otherstate)<1e-10;
            elseif contraction_path=="full"
                amplitude,sample0, _=contract_sample(psi_decomposed,Lx,Ly,iconf_new,sample0, Vp,contract_fun);
            elseif contraction_path=="recycle"
                amplitude,sample0, _,contract_history= partial_contract_sample(psi_decomposed,iconf_new,sample0, Vp,contract_history);
                amplitude_otherstate,sample1,_,contract_history_otherstate= partial_contract_sample(psi_decomposed_otherstate,iconf_new,sample1,Vp,contract_history_otherstate);
            end

            for j in 1:Nbra  # Inner loop to create uncorrelated samples
                randl = rand(1:L)  # Picking a site at random; "l"
                rand2 = rand(1:length(NN_tuple[randl]))  # Picking randomly one of the 4 neighbors
                randK = NN_tuple[randl][rand2]  # Picking a neighbor at random to which electron wants to hop; "K"

                

                if iconf_new[randl] != iconf_new[randK]
                    

                    iconf_new_flip=flip_config(iconf_new,randl,randK);
                    if contraction_path=="verify"
                        # println(iconf_new)
                        # pos1=findall(x->x.==randl,reshape(coord,Lx,Ly));
                        # pos2=findall(x->x.==randK,reshape(coord,Lx,Ly));
                        # @show pos1
                        # @show pos2
                        amplitude_flip,sample0, _=contract_sample(psi_decomposed,Lx,Ly,iconf_new_flip,sample0, Vp,contract_fun);
                        amplitude_flip_,sample0, _,contract_history_flip= partial_contract_sample(psi_decomposed,iconf_new_flip,sample0, Vp,contract_history);
                        @assert abs((amplitude_flip-amplitude_flip_)/amplitude_flip_)<1e-10   string(amplitude_flip)*", "*string(amplitude_flip_);
                    elseif contraction_path=="full"
                        amplitude_flip,sample0, _=contract_sample(psi_decomposed,Lx,Ly,iconf_new_flip,sample0, Vp,contract_fun);
                    elseif contraction_path=="recycle"
                        amplitude_flip,sample0, _, contract_history_flip= partial_contract_sample(psi_decomposed,iconf_new_flip,sample0, Vp,contract_history);
                    end

                    probratio = abs2(amplitude_flip/amplitude)  # Probability of accepting configuration
                    eta_rand = rand()  # Random number from 0 to 1; "eta"

                    if eta_rand < probratio  # We accept the configuration
                        # println("accept")
                        iconf_new= iconf_new_flip;
                        amplitude=amplitude_flip;
                        if contraction_path in ("verify","recycle")
                            contract_history=contract_history_flip;
                        end
                    end
                end
            end
            
            energyl1,sample1,contract_history_otherstate= overlap_ratio(iconf_new,sample1,amplitude, contract_history_otherstate)

            rems = mod1(i, binn)  # Binning to store fewer numbers, usually binn is order of 1000
            ebin1[rems] = energyl1

            if rems == binn
                #println( mean(ebin1))
                #CSV.write(outputname, mean(ebin1); append=true) 
                println(file, mean(ebin1));flush(file);
            end

            # Optional: Uncomment to print configuration every 999 steps
            #save a good initial config for next time
            if mod(i + 1, 999) == 0
                # println(outfile, "\n\n", iconf_new, "\n\n\n")
                # println(file, "\n\n", iconf_new, "\n\n\n");flush(stdout);
            end

            # if mod(i + 1, 200) == 0
            # println(mean(ebin1));flush(stdout);
            # end

        end

    end
    GC.gc(true);

end


for BC1=1:4;
    for BC2=1:4;
        @show BC1
        @show BC2
        if BC1==BC2
            continue;
        end

        ntask=nworkers();
        # main(dir, 1, ntask, BC1,BC2);
        @sync begin
            for cp=1:ntask
                worker_id=workers()[cp]
                @spawnat worker_id main(dir, worker_id, ntask, BC1,BC2);
            end
        end

        data_set=Vector{ComplexF64}(undef,0);
        for cp =1:ntask
            outputname = dir*"id_"*string(workers()[cp])*"_BC"*string(BC1)*"_"*string(BC2)*"_chi"*string(chi)*".csv";
            # Read the list of numbers from the CSV file
            data = open(outputname, "r") do file
                [parse(ComplexF64, line) for line in readlines(file)]
            end
            data_set=vcat(data_set,data);
            rm(outputname)
        end



        #error analysis



        begin
            # Output file
            
            bin_size_set=Vector{Int}(undef,0);
            energy_set=Vector{ComplexF64}(undef,0); 
            std_dev_set=Vector{Float64}(undef,0);

            bin_size = 1

            total_data_size = length(data_set)
            while bin_size < total_data_size
                # Bin the data
                binned = [mean(data_set[i:min(i+bin_size-1, total_data_size)]) for i in 1:bin_size:total_data_size]
                
                # Compute mean energy per site
                energy = mean(binned) 
                
                # Compute standard deviation
                std_dev = std(binned; corrected=false) / (sqrt(length(binned)))

                # Write results to the output file
                push!(bin_size_set,bin_size);
                push!(energy_set,energy);
                push!(std_dev_set,std_dev);

                # Double the bin size
                bin_size *= 2
            end

            matnm=string(Lx)*"x"*string(Ly)*"_D"*string(D)*"_BC"*string(BC1)*"_"*string(BC2)*"_chi"*string(chi)*".mat"
            matwrite(matnm, Dict(
            "bin_size_set"=>bin_size_set,
            "energy_set"=>energy_set,
            "std_dev_set"=>std_dev_set,
            "data_set"=>data_set
            ); compress = false)    
        end 

    end
end

    



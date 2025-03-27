using Distributed
using LinearAlgebra:I,diagm,diag
using TensorKit
using Random
using Printf
using DelimitedFiles
using JLD2
using Dates


cd(@__DIR__)


include("src/Settings.jl")
include("src/contract_torus.jl")
include("src/sampling.jl")
include("src/sampling_eliminate_physical_leg.jl")


#set parameters
const Lx = 6      # number of sites along x / number of columns in the lattice
const Ly = 6      # number of sites along y / number of rows in the lattice
const D=3;#bond dimension of state
const chi=10;#bond dimension of environment
const L = Lx * Ly # total number of lattice sites
const Nbra = L             # Inner loop size, to generate uncorrelated samples, usually must be of size O(L).
const Nsteps = 400000       # Total Monte Carlo steps
const binn = 1000          # Bin size to store the data during the monte carlo run. 
const GC_spacing = 200          # garbage collection


####################
#use single core
import LinearAlgebra.BLAS as BLAS
using Base.Threads

n_cpu=1;
BLAS.set_num_threads(n_cpu);
####################

#restrict size of cache:
# TensorKit.usebraidcache_abelian[] = false 
# TensorKit.usebraidcache_nonabelian[] = false
TensorKit.braidcache.maxsize=1000
TensorKit.transposecache.maxsize=1000
# TensorKit.usetransposecache
TensorKit.treepermutercache.maxsize=1000
TensorKit.GLOBAL_FUSIONBLOCKSTRUCTURE_CACHE.maxsize=1000
# Base.summarysize(TensorKit.treepermutercache)
# Base.summarysize(TensorKit.GLOBAL_FUSIONBLOCKSTRUCTURE_CACHE)

function meminfo_julia()
    # @printf "GC total:  %9.3f MiB\n" Base.gc_total_bytes(Base.gc_num())/2^20
    # Total bytes (above) usually underreports, thus I suggest using live bytes (below)
    @printf "GC live:   %9.3f MiB\n" Base.gc_live_bytes()/2^20
    @printf "JIT:       %9.3f MiB\n" Base.jit_total_bytes()/2^20
    @printf "Max. RSS:  %9.3f MiB\n" Sys.maxrss()/2^20
end

function main()
    #load U(1) symmetric tensor and then produce PEPS on a finite cluster
    filenm="CSL_D"*string(D)*"_U1";
    psi0,Vp,Vv=load_fPEPS_from_iPEPS(Lx,Ly,filenm,false);
    global psi_decomposed, Vp
    normalize_PEPS!(psi0,Vp,contract_whole_torus_boundaryMPS);#normalize psi0 
    psi_decomposed=decompose_physical_legs(psi0,Vp);
    sample0=Matrix{TensorMap}(undef,Lx,Ly);
    ##########################################
    coord,fnn_set,snn_set,NN_tuple,NNN_tuple, NN_tuple_reduced,NNN_tuple_reduced=get_neighbours(Lx,Ly,"PBC");

    #create empty contract_history
    contract_history=torus_contract_history(zeros(Int8,Lx*Ly),Matrix{TensorMap}(undef,Lx,Ly));
    # Initialize variables
    iconf_new =initial_Neel_config(Lx,Ly,1);


    starting_time=now();
    for i in 1:Nsteps  # Number of Monte Carlo steps, usually 1 million
        global ite_num
        ite_num=i;

        #contract PEPS samples
        amplitude,sample0, _,contract_history= partial_contract_sample(psi_decomposed,iconf_new,sample0, Vp,contract_history);
        for j in 1:Nbra  # Inner loop to create uncorrelated samples
            randl = rand(1:L)  # Picking a site at random; "l"
            rand2 = rand(1:length(NN_tuple[randl]))  # Picking randomly one of the 4 neighbors
            randK = NN_tuple[randl][rand2]  # Picking a neighbor at random to which electron wants to hop; "K"
            iconf_new=flip_config(iconf_new,randl,randK);
            amplitude,sample0, _, contract_history= partial_contract_sample(psi_decomposed,iconf_new,sample0, Vp,contract_history);
        end
        

        #collect garbage and print memory cost
        if mod(i, 100) == 0
            GC.gc(true);
            meminfo_julia();
            Now=now();
            Time=Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(Now) - Dates.DateTime(starting_time)));
            println("Time consumed: "*string(Time));flush(stdout);
        end

    end

end


 main();
    

    



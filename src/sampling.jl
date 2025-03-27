function flip_config(config0::Vector,pos1::Int,pos2::Int)
    config=deepcopy(config0);
    config[pos1]=config0[pos2];
    config[pos2]=config0[pos1];
    return config
end

function load_fPEPS_from_iPEPS(Lx::Int,Ly::Int,filenm::String,to_dense)
    data=load(filenm*".jld2");
    A=data["A"];

    # A=TensorMap(A.data,A.codom,A.dom);

    if to_dense
        A_dense=convert(Array,A);
        siz=size(A_dense);
        A=TensorMap(A_dense, ℂ^siz[1]*ℂ^siz[2]*(ℂ^siz[3])'*(ℂ^siz[4])', ℂ^siz[5]);
        Vv=space(A,1);
        Vp=space(A,5);
    else
        Vv=U₁Space(0=>1,1/2=>1,-1/2=>1);
        Vp=U₁Space(1/2=>1,-1/2=>1);
        # Vv=ℤ₂Space(0=>1,1=>2);
        # Vp=ℤ₂Space(1=>2);
        A=TensorMap(convert(Array,A),Vv*Vv,Vv*Vv*Vp);
    end
    

    A=permute(A,(1,2,3,4,5,));
    
    
    #psi=generate_obc_from_iPEPS(A,Lx,Ly);
    psi=Matrix{TensorMap}(undef,Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            psi[cx,cy]=A;
        end
    end
    return psi,Vp,Vv
end





function get_neighbours(Lx,Ly,boundary_condition)
    #determine neighbours
    coord=reshape(Vector(1:Lx*Ly),(Lx,Ly));
    fnn_set=zeros(Int,Lx*Ly);
    snn_set=zeros(Int,Lx*Ly);
    NN_matrix=zeros(Int,Lx*Ly,4);
    NNN_matrix=zeros(Int,Lx*Ly,4);

    nn=[1 0;-1 0;0 1;0 -1];
    nnn=[1 1;1 -1;-1 1;-1 -1];

    for px in 1:Lx
        for py in 1:Ly
            NN=[];
            NNN=[];
            if boundary_condition in ("OBC",)
                for cn =1:4
                    if (px+nn[cn,1] in 1:Lx) && (py+nn[cn,2] in 1:Ly)
                        push!(NN,coord[px+nn[cn,1], py+nn[cn,2]]);
                    end
                    if (px+nnn[cn,1] in 1:Lx) && (py+nnn[cn,2] in 1:Ly)
                        push!(NNN,coord[px+nnn[cn,1], py+nnn[cn,2]]);
                    end
                end
            elseif boundary_condition in ("PBC",)
                for cn =1:4
                    push!(NN,coord[mod1(px+nn[cn,1],Lx), mod1(py+nn[cn,2],Ly)]);
                    push!(NNN,coord[mod1(px+nnn[cn,1],Lx), mod1(py+nnn[cn,2],Ly)]);
                end
            end
            fnn_set[coord[px,py]]=length(NN);
            snn_set[coord[px,py]]=length(NNN);
            NN_matrix[coord[px,py],1:length(NN)]=NN;
            NNN_matrix[coord[px,py],1:length(NNN)]=NNN;
            
        end
    end

    
    function neighbour_convert_to_tuple(Lx,Ly,M)
        M_tuple=Vector{Tuple}(undef,Lx*Ly);
        for c1=1:Lx*Ly
            pos=findall(x->x.>0, M[c1,:]);
            M_tuple[c1]=Tuple(M[c1,pos]);
        end
        return M_tuple
    end

    NN_tuple=neighbour_convert_to_tuple(Lx,Ly,NN_matrix);
    NNN_tuple=neighbour_convert_to_tuple(Lx,Ly,NNN_matrix);
    # @show NN_matrix
    
    NN_matrix_reduced=deepcopy(NN_matrix);
    NNN_matrix_reduced=deepcopy(NNN_matrix);
    #remove double counting
    for c1=1:Lx*Ly
        for c2=1:size(NN_matrix_reduced,2)
            p1=NN_matrix_reduced[c1,c2];
            if p1>0
                if c1 in NN_matrix_reduced[p1,:]
                    NN_matrix_reduced[c1,c2]=0
                end
            end
        end
        for c2=1:size(NNN_matrix_reduced,2)
            p1=NNN_matrix_reduced[c1,c2];
            if p1>0
                if c1 in NNN_matrix_reduced[p1,:]
                    NNN_matrix_reduced[c1,c2]=0
                end
            end
        end
    end


    NN_tuple_reduced=neighbour_convert_to_tuple(Lx,Ly,NN_matrix_reduced);
    NNN_tuple_reduced=neighbour_convert_to_tuple(Lx,Ly,NNN_matrix_reduced);

    return coord,fnn_set,snn_set,NN_tuple,NNN_tuple, NN_tuple_reduced,NNN_tuple_reduced
end


function initial_Neel_config(Lx,Ly,sign)
    #initial spin config, total sz=0
    @assert sign in (1,-1);
    config=zeros(Int8,Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            config[cx,cy]=(-1)^(cx+cy)*sign;
        end
    end
    @assert sum(sum(config))==0
    return config[:]
end


function decompose_physical_legs(fPEPS0::Matrix{TensorMap},Vp::GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
    
    Lx,Ly=size(fPEPS0);
    fPEPS=Array{TensorMap}(undef,Lx,Ly,TensorKit.dim(Vp));
    Vp=U₁Space(1/2=>1,-1/2=>1);
    Vup=U₁Space(1/2=>1);
    Vdn=U₁Space(-1/2=>1);
    Pup=TensorMap([1,0]',Vup',Vp');
    Pdn=TensorMap([0,1]',Vdn',Vp');
    projectors=[Pup,Pdn];
    @assert TensorKit.dim(Vp)==2;
    for cp=1:TensorKit.dim(Vp)
        for cx=1:Lx
            for cy=1:Ly
                T=fPEPS0[cx,cy];
                @tensor T[:]:=T[-1,-2,-3,-4,1]*projectors[cp][-5,1];
                fPEPS[cx,cy,cp]=T;
            end
        end
    end
    return fPEPS
end



function apply_sampling_projector(fPEPS::Matrix{TensorMap},config::Matrix,sample::Matrix{TensorMap}, Vp::GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
    fPEPS=deepcopy(fPEPS);
    Lx,Ly=size(fPEPS);
    Vp=U₁Space(1/2=>1,-1/2=>1);
    Vup=U₁Space(1/2=>1);
    Vdn=U₁Space(-1/2=>1);
    Pup=TensorMap([1,0]',Vup',Vp');
    Pdn=TensorMap([0,1]',Vdn',Vp');

    for cx=1:Lx
        for cy=1:Ly
            T=fPEPS[cx,cy];
            if config[cx,cy]==1
                @tensor T[:]:=T[-1,-2,-3,-4,1]*Pup[-5,1];
            elseif config[cx,cy]==-1
                @tensor T[:]:=T[-1,-2,-3,-4,1]*Pdn[-5,1];
            end
            fPEPS[cx,cy]=T;
        end
    end
    return fPEPS
end





function apply_sampling_projector(fPEPS,Lx::Int,Ly::Int,config::Vector,sample,Vp)
    return apply_sampling_projector(fPEPS,reshape(config,Lx,Ly),sample,Vp)
end


function pick_sample(fPEPS_decomposed::Array{TensorMap},config0::Vector, sample::Matrix{TensorMap})
    Lx,Ly,Lp=size(fPEPS_decomposed);
    config=reshape(config0,Lx,Ly)
    @assert Lp==2;#spin model
    # fPEPS=Matrix{TensorMap}(undef,Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            if config[cx,cy]==1
                sample[cx,cy]=fPEPS_decomposed[cx,cy,1];
            elseif config[cx,cy]==-1
                sample[cx,cy]=fPEPS_decomposed[cx,cy,2];
            end
        end
    end
    return sample
end

function normalize_PEPS!(psi::Matrix{TensorMap},Vp,contract_fun::Function)
    Lx,Ly=size(psi);
    config=initial_Neel_config(Lx,Ly,1);
    psi_sample=Matrix{TensorMap}(undef,Lx,Ly);
    psi_sample=apply_sampling_projector(psi,Lx,Ly,config,psi_sample, Vp);
    if isa(Vp,GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
        psi_sample=shift_pleg(psi_sample);
    end
    chi__=30;
    Norm,trun_err=contract_fun(psi_sample,chi__);
    Norm=norm(Norm);
    coe=Norm^(1/(Lx*Ly));
    for cc in eachindex(psi)
        setindex!(psi,psi[cc]/coe,cc);
    end
end



function contract_sample(psi::Matrix{TensorMap},Lx::Int,Ly::Int,config::Vector,psi_sample_old::Matrix{TensorMap}, Vp::GradedSpace,contract_fun::Function)
    psi_sample=apply_sampling_projector(psi,Lx,Ly,config,psi_sample_old, Vp);
    if isa(Vp,GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
        psi_sample=shift_pleg(psi_sample);
    end
    Norm,trun_err=contract_fun(psi_sample,chi);
    return Norm,psi_sample, trun_err
end

function contract_sample(psi_decomposed::Array{TensorMap},Lx::Int,Ly::Int,config::Vector,psi_sample_old, Vp::GradedSpace,contract_fun::Function)
    psi_sample=pick_sample(psi_decomposed,config, psi_sample_old);
    if isa(Vp,GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
        psi_sample=shift_pleg(psi_sample);
    end
    # Norm,trun_err,m,n=contract_fun(psi_sample,chi);
    # return Norm,trun_err,m,n
    Norm,trun_err=contract_fun(psi_sample,chi);
    return Norm,psi_sample, trun_err
end




function partial_contract_sample(psi_decomposed::Array{TensorMap},config::Vector,psi_sample_old::Matrix{TensorMap}, Vp::GradedSpace,contract_history_::torus_contract_history)
    psi_sample=pick_sample(psi_decomposed,config, psi_sample_old);
    if isa(Vp,GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
        psi_sample=shift_pleg(psi_sample);
    end
    Norm,trun_errs, contract_history_new=contract_partial_torus_boundaryMPS(psi_sample,config,contract_history_, chi)

    #################################
    #for verification, need to comment later
    # jldsave("test2.jld2";psi_decomposed,config,contract_history_,contract_history_new,chi)
    # verify_contract_history(psi_sample,contract_history_new, chi);
    #################################
    global ite_num
    if mod(ite_num,GC_spacing)==0
        GC.gc(true);
    end
    return Norm,psi_sample, trun_errs, contract_history_new
end

function partial_contract_sample(psi_decomposed::Array{TensorMap},config::Vector,psi_sample_old::Matrix{TensorMap}, Vp::ComplexSpace,contract_history_::torus_contract_history)
    psi_sample=pick_sample(psi_decomposed,config, psi_sample_old);

    Norm,trun_errs, contract_history_new=contract_partial_torus_boundaryMPS(psi_sample,config,contract_history_, chi)

    #################################
    #for verification, need to comment later
    # jldsave("test2.jld2";psi_decomposed,config,contract_history_,contract_history_new,chi)
    # verify_contract_history(psi_sample,contract_history_new, chi);
    #################################
    global ite_num
    if mod(ite_num,GC_spacing)==0
        GC.gc(true);
    end
    return Norm,psi_sample, trun_errs, contract_history_new
end
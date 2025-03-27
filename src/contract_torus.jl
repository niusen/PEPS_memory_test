function anti_clock_coord_rotate(coord,Lx,Ly)
    return [Ly-coord[2]+1,coord[1]]
end


function combine_two_rows_method1(mpo1::Vector{TensorMap},mpo2::Vector{TensorMap},chi::Int)
    function build_projector(Tup::TensorMap,Tdn::TensorMap,chi::Int)
        Tup=permute(Tup,(1,2,3,4,));
        Tdn=permute(Tdn,(1,2,3,4,));
        # @tensor TTTT[:]:=Tup[-1,1,-4,-6]*Tdn[-2,-3,-5,1];
        @tensor TTTT[:]:=Tup[-1,5,2,1]*Tup'[-3,6,2,1]*Tdn[-2,4,3,5]*Tdn'[-4,4,3,6];
        TTTT=permute(TTTT,(1,2,),(3,4,));
        uM,sM,vM = tsvd(TTTT; trunc=truncdim(chi))
        err=norm(uM*sM*vM-TTTT)/norm(TTTT);
        # println(err)
        # println(eu)
    
    
        PR=uM;
        PL=uM';
        return PR,PL,err
    end
    trun_err=Vector{Float64}(undef,0);
    L=length(mpo1);
    local PL_set=Vector{TensorMap}(undef,L);
    local PR_set=Vector{TensorMap}(undef,L);
    for cc=1:L
        PR,PL,err=build_projector(mpo1[cc],mpo2[cc],chi);#projector at left side
        trun_err=vcat(trun_err,err);
        PL_set[cc]=PL;
        PR_set[mod1(cc-1,L)]=PR;
    end
    local mpo_new=Vector{TensorMap}(undef,L);
    for cc=1:L
        Tup=mpo1[cc];
        Tdn=mpo2[cc];
        PL=PL_set[cc];
        PR=PR_set[cc];
        @tensor Tnew[:]:=PL[-1,1,3]*Tup[1,2,4,-4]*Tdn[3,-2,5,2]*PR[4,5,-3]; #the cost of this step may be D^7 or chi^7
        mpo_new[cc]=Tnew;
    end

    PL_set=[];
    PR_set=[];
    return mpo_new,trun_err
end

function contract_rows(finite_PEPS::Matrix{TensorMap},chi::Int)
    global projector_method
    if projector_method=="1"
        combine_two_rows=combine_two_rows_method1
    elseif projector_method=="2"
        combine_two_rows=combine_two_rows_method2
    elseif projector_method=="3"
        combine_two_rows=combine_two_rows_method3
    end
    Lx,Ly=size(finite_PEPS);
    local finite_PEPS_=deepcopy(finite_PEPS);
    if mod(Ly,2)==0
        Ly_new=Int(Ly/2);
    else
        Ly_new=Int((Ly+1)/2);
    end
    trun_err=Vector{Float64}(undef,0);

    local PEPS_new=Matrix{TensorMap}(undef,Lx,Ly_new);
    for cy=1:2:Ly
        if cy<Ly
            mpo1=finite_PEPS_[:,cy+1];
            mpo2=finite_PEPS_[:,cy];
            mpo_new,err=combine_two_rows(mpo1,mpo2,chi);
            # trun_err=vcat(trun_err,err);
            PEPS_new[:,Int((cy+1)/2)]=mpo_new;
        elseif cy==Ly #Ly odd
            PEPS_new[:,Ly_new]=finite_PEPS_[:,cy];
        end
    end
    finite_PEPS_=[];
    return PEPS_new,trun_err
end


# function contract_whole_torus(finite_PEPS::Matrix{TensorMap},chi::Int)
#     finite_PEPS=deepcopy(finite_PEPS);
#     trun_err=Vector{Float64}(undef,0);
#     Ly_=size(finite_PEPS,2);
#     global rotate_truncation

#     if rotate_truncation==false
#         while Ly_>2
#             finite_PEPS,err=contract_rows(finite_PEPS,chi)
#             trun_err=vcat(trun_err,err);
#             Lx,Ly_=size(finite_PEPS);
#             # if Ly_==2
#             #     break;
#             # end
#         end

#         ###################################
#         #final contraction method 1
#         Lx,Ly_=size(finite_PEPS);
#         cc=1;
#         @tensor T[:]:=finite_PEPS[cc,2][-1,1,-3,2]*finite_PEPS[cc,1][-2,2,-4,1];
#         T=permute(T,(1,2,),(3,4,));
#         for cc=2:Lx-1
#             @tensor T_[:]:=finite_PEPS[cc,2][-1,1,-3,2]*finite_PEPS[cc,1][-2,2,-4,1];
#             T_=permute(T_,(1,2,),(3,4,));
#             T=T*T_;
#         end
#         cc=Lx;
#         @tensor T_[:]:=finite_PEPS[cc,2][-1,1,-3,2]*finite_PEPS[cc,1][-2,2,-4,1];
#         ov=@tensor T[1,2,3,4]*T_[3,4,1,2]
#         #############################
#         # #final contraction method 2, slower than method1
#         # Lx,Ly_=size(finite_PEPS);
#         # cc=1;
#         # @tensor T[:]:=finite_PEPS[cc,2][-1,1,-3,2]*finite_PEPS[cc,1][-2,2,-4,1];
#         # for cc=2:Lx-1
#         #     @tensor T[:]:=T[-1,-2,4,1]*finite_PEPS[cc,2][4,2,-3,3]*finite_PEPS[cc,1][1,3,-4,2];
#         # end
#         # cc=Lx;
#         # ov=@tensor  T[5,2,3,1]*finite_PEPS[cc,2][3,4,5,6]*finite_PEPS[cc,1][1,6,2,4];
#         #############################
#     else
#         while Ly_>2
#             finite_PEPS_new,err=contract_rows(finite_PEPS,chi)
#             trun_err=vcat(trun_err,err);
#             Lx,Ly_=size(finite_PEPS_new);
#             #clockwise rotate
#             finite_PEPS=Matrix{TensorMap}(undef,Ly_,Lx);
#             for cx =1:Lx
#                 for cy=1:Ly_
#                     coord_new=anti_clock_coord_rotate([cx,cy],Lx,Ly_);
#                     finite_PEPS[coord_new[1],coord_new[2]]=permute(finite_PEPS_new[cx,cy],(4,1,2,3,));
#                 end
#             end
#             Lx,Ly_=size(finite_PEPS);

#         end

#         ###################################
#         #final contraction method 1
#         Lx,Ly_=size(finite_PEPS);
#         cc=1;
#         @tensor T[:]:=finite_PEPS[cc,2][-1,1,-3,2]*finite_PEPS[cc,1][-2,2,-4,1];
#         T=permute(T,(1,2,),(3,4,));
#         for cc=2:Lx-1
#             @tensor T_[:]:=finite_PEPS[cc,2][-1,1,-3,2]*finite_PEPS[cc,1][-2,2,-4,1];
#             T_=permute(T_,(1,2,),(3,4,));
#             T=T*T_;
#         end
#         cc=Lx;
#         @tensor T_[:]:=finite_PEPS[cc,2][-1,1,-3,2]*finite_PEPS[cc,1][-2,2,-4,1];
#         ov=@tensor T[1,2,3,4]*T_[3,4,1,2]

#     end

#     finite_PEPS=[];
#     return ov,trun_err
# end



function contract_whole_torus_boundaryMPS(finite_PEPS::Matrix{TensorMap},chi::Int)
    finite_PEPS=deepcopy(finite_PEPS);
    Lx,Ly=size(finite_PEPS);
    trun_err=Vector{Float64}(undef,0);
    

    combine_two_rows=combine_two_rows_method1


    # @time begin
    ppy=Int(round(Ly/2));

    mpo_bot=finite_PEPS[:,1];
    for cy=2:ppy
        mpo_bot,err_set=combine_two_rows(finite_PEPS[:,cy],mpo_bot,chi);  
        trun_err=vcat(trun_err,err_set);  
    end


    ###################################
    # mpo_top=finite_PEPS[:,ppy+1];
    # for cy=ppy+2:Ly
    #     mpo_top,err_set=combine_two_rows(finite_PEPS[:,cy],mpo_top,chi);  
    #     trun_err=vcat(trun_err,err_set);  
    # end
    ###################################
    mpo_top=finite_PEPS[:,Ly];
    for cy=Ly-1:-1:ppy+1
        mpo_top,err_set=combine_two_rows(mpo_top,finite_PEPS[:,cy],chi);  
        trun_err=vcat(trun_err,err_set);  
    end

    # end
    # @time begin
    ###################################
    #final contraction method 1
    Lx_,Ly_=size(finite_PEPS);
    cc=1;
    @tensor T[:]:=mpo_top[cc][-1,1,-3,2]*mpo_bot[cc][-2,2,-4,1];
    T=permute(T,(1,2,),(3,4,));
    for cc=2:Lx_-1
        @tensor T_[:]:=mpo_top[cc][-1,1,-3,2]*mpo_bot[cc][-2,2,-4,1];
        T_=permute(T_,(1,2,),(3,4,));
        T=T*T_;
    end
    cc=Lx_;
    @tensor T_[:]:=mpo_top[cc][-1,1,-3,2]*mpo_bot[cc][-2,2,-4,1];
    ov=@tensor T[1,2,3,4]*T_[3,4,1,2]
    #############################
    # end
    finite_PEPS=[];
    return ov,trun_err
end



function contract_partial_torus_boundaryMPS(psi_single::Matrix{TensorMap},config_new::Vector{Int8},contract_history_::torus_contract_history,chi::Int)
    local psi_single_=deepcopy(psi_single);
    contract_history_=deepcopy(contract_history_);#warning: this deepcopy is necessary, otherwise may cause error is sweep is not accepted.
    Lx,Ly=size(psi_single_);#original cluster size without adding trivial boundary
    config_new_=reshape(config_new,Lx,Ly);
    config_old_=reshape(contract_history_.config,Lx,Ly);
    ppy=Int(round(Ly/2));


        combine_two_rows=combine_two_rows_method1


    #compare old and new config
    y_bot0=0;
    for cy=1:ppy
        if config_new_[:,cy]==config_old_[:,cy]
            y_bot0=y_bot0+1;
        else
            break;
        end
    end

    y_top0=Ly+1;
    for cy=Ly:-1:ppy+1
        if config_new_[:,cy]==config_old_[:,cy]
            y_top0=y_top0-1;
        else
            break;
        end
    end
    # @show y_bot0,y_top0
    

    ##########################
    trun_err=Vector{Float64}(undef,0);
    local mps_all_set=contract_history_.mps_all_set;

    # @time begin
 
    
    if y_bot0==0
        mps_bot=psi_single_[:,1];
        mps_all_set[:,1]=mps_bot;
        y0=1;
    elseif y_bot0>0
        mps_bot=mps_all_set[:,y_bot0];
        y0=y_bot0;
    end

    for cy=y0+1:ppy
        mps_bot,trun_errs=combine_two_rows(psi_single_[:,cy], mps_bot,chi);
        mps_all_set[:,cy]=mps_bot;
        # trun_err=vcat(trun_err,trun_errs);
    end

    #######################


    if y_top0==Ly+1
        mps_top=psi_single_[:,Ly];
        mps_all_set[:,Ly]=mps_top;
        y1=Ly;
    elseif y_top0<Ly+1
        mps_top=mps_all_set[:,y_top0];
        y1=y_top0;
    end

    
    
    for cy=y1-1:-1:ppy+1
        mps_top,trun_errs=combine_two_rows(mps_top,psi_single_[:,cy],chi);
        mps_all_set[:,cy]=mps_top;
        # trun_err=vcat(trun_err,trun_errs);
    end



    # end
    # @time begin
    ###################################

    #final contraction method 1
    cc=1;
    @tensor T[:]:=mps_top[cc][-1,1,-3,2]*mps_bot[cc][-2,2,-4,1];
    T=permute(T,(1,2,),(3,4,));
    for cc=2:Lx-1
        @tensor T_[:]:=mps_top[cc][-1,1,-3,2]*mps_bot[cc][-2,2,-4,1];
        T_=permute(T_,(1,2,),(3,4,));
        T=T*T_;
    end
    cc=Lx;
    @tensor T_[:]:=mps_top[cc][-1,1,-3,2]*mps_bot[cc][-2,2,-4,1];
    ov=@tensor T[1,2,3,4]*T_[3,4,1,2]
    #############################


    # end
    local torus_contract_history_new=torus_contract_history(config_new, mps_all_set);
    psi_single_=[];
    contract_history_=[];
    mps_bot=[];
    mps_top=[];
    mps_all_set=[];
    global ite_num
    if mod(ite_num,GC_spacing)==0
        GC.gc(true);
    end
    return ov, trun_err,  torus_contract_history_new
end


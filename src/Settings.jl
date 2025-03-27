function Rank(T::TensorMap)
    #number of indices
    return length((domain(T)*codomain(T)).spaces)
end 


abstract type Contract_History end



struct torus_contract_history <:Contract_History
    config :: Vector;
    mps_all_set::Matrix{TensorMap}
end


function convert_to_dense(T::Tensor)
    function convert_to_dense_space(V)
        if V.dual
            return ComplexSpace(dim(V))';
        else
            return ComplexSpace(dim(V));
        end
    end

    if Rank(T)==5;
        T=permute(T,(1,2,3,4,5,));
        V1=convert_to_dense_space(space(T,1));
        V2=convert_to_dense_space(space(T,2));
        V3=convert_to_dense_space(space(T,3));
        V4=convert_to_dense_space(space(T,4));
        V5=convert_to_dense_space(space(T,5));
        T_dense=convert(Array,T);
        T_new=TensorMap(T_dense,V1*V2*V3*V4,V5');
        T_new=permute(T_new,(1,2,3,4,5,))
    end
    return T_new
end

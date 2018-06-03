module Utils
export crandn, nrows, ncols, supp, row_supp, large_ind

function crandn(dims...)
    sqrt(0.5) * ( randn(dims) + 1im*randn(dims) )
end
function nrows(A::AbstractArray)
    return size(A,1)
end
function ncols(A::AbstractArray)
    return size(A,2)
end
function supp(x)
	return find(x .!= 0)
end
function row_supp(x::Matrix)
	ncols(x) == 1 && return supp(vec(x))
	return union(find(x[:,1] .!= 0), row_supp(x[:,2:end]))
end
function large_ind(x;k=1::Int)
	return sortperm(x,rev=true)[1:k]
end

end

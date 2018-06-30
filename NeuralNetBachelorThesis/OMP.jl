#module OMP
function OMP(s,d,m,N,phi)
#phi = rand(d,N);
#phi = eye(d,d)
phi = transpose(phi);
#phi = eye(d)
v = phi*s;
#initialize residual, index set and counter
r_t = v; t = 1; i_t = zeros(Int64,m); res = zeros(Complex128,d);
phi_t=zeros(size(phi)...); 
if(m>d)
    x = zeros(Complex128, m);
else
    x = zeros(Complex128, d);
end
while(t<=m)
    #solve lambda = arg max|<r_t,phi(:,k)>
    max = 0; lambda =0;
    for k=1:d
        #product = abs(vecdot(r_t,phi[:,k]))/norm(phi[:,k]);
        product = abs(transpose(conj(phi[:,k]))*r_t)/norm(phi[:,k]);
        if product >= max
            max = product;
            lambda = k;
        end
    end
    #Augment index set and phi_t
    i_t[t] = lambda;
    phi_t[:,t] = phi[:,lambda];
    #solve least squares problem: x=arg min||v-phi_t*x||
    x[1:t] = pinv(phi_t[:,1:t])*v;
    #calculate new approximation of data and new residual
    a_t = phi_t[:,1:t]*x[1:t];
    r_t = v - a_t;
    #increment counter
    t = t+1;
end

for i in 1:m
    res[i_t[i]] = x[i]
end
#return countnz(s.*sparsevec(i_t,x[1:m],d))/m;
return (res, i_t)
end

function sortOMP(s,m)
    return (s[sortperm(abs(s))[1:m]],sortperm(abs(s))[1:m])
end

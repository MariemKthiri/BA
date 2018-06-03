module OMP
function OMP(s,d,m,N)
phi = rand(d,N);
phi = transpose(phi);
v = phi*s;
#initialize residual, index set and counter
r_t = v; t = 1; i_t = zeros(m)
phi_t=zeros(N,d); x = zeros(max(m,N));
while(t<=m)
    #solve lambda = arg max|<r_t,phi(:,k)>
    max = 0; lambda =0;
    for k=1:d
        product = abs(vecdot(r_t,phi[:,k]))/norm(phi[:,k]);
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
#return countnz(s.*sparsevec(i_t,x[1:m],d))/m;
return x
end
end
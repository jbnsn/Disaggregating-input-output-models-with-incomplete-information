% Source: https://www.tandfonline.com/doi/suppl/10.1080/09535314.2012.689954

% Lindner, Sören, Julien Legault, and Dabo Guan. 2012.
% ‘Disaggregating Input–Output Models with Incomplete Information’. Economic Systems Research 24 (4): 329–47.
% https://doi.org/10.1080/09535314.2012.689954.

%============
%Loading data
%============
load('IOT_China.mat'); %Loading China's IO table
x = IOT_national(:,end); %Vector of total outputs
f = IOT_national(:,end-1); %Vector of final demand
id = IOT_national(:,end-2); %Vector of intermediate demand
Z = IOT_national(:,1:end-3); %Exchange matrix
temp = size(Z); %Size of IO table
N = temp(1)-1; %Number of common sectors
A = Z./repmat(transpose(x),N+1,1); %Aggregated technical coefficient matrix
x_common = x(1:end-1); %Vector of total outputs for common sectors
f_common = f(1:end-1); %Vector of final demand for common sectors
%Note: The last sector of the table is disaggregated, i.e. the electricity sector
x_elec = x(end); %Total output of the disaggregated sector
f_elec = f(end); %Final demand of the disaggregated sector
 
%---Newly formed sectors from the electricity sector---%
n = 3; %Number of new sectors
w = [0.241;0.648;0.111]; %New sector weights
N_tot = N + n; %Total number of sectors for the disaggregated IO table
x_new = w.*x_elec; %Vector of new total sector outputs
xs = [x_common;x_new]; %Vector of disaggregated economy sector total outputs
f_new = w*f_elec; %Final demand of new sectors
 
%================================
%Building the constraint matrix C
%================================
Nv = n*N_tot + n; %Number of variables
Nc = N + n + 1; %Number of constraints
q = [transpose(A(N+1,:));w]; %Vector of constraint constants
C = zeros(Nc,Nv); %Matrix of constraints
 
%---Common sectors constraints---%
C11 = zeros(N,N*n);
for ii = 1:N
    col_indices = n*(ii-1)+1:n*ii;
    C11(ii,col_indices) = ones(1,n);
end
C(1:N,1:N*n) = C11;
 
%---New sectors constraints---%
C22 = zeros(1,n^2);
for ii = 1:n
    col_indices = n*(ii-1)+1:n*ii;
    C22(1,col_indices) = w(ii)*ones(1,n);
end
C(N+1,N*n+1:N*n+n^2) = C22;
 
%---Final demand constraints---%
C31 = zeros(n,N*n);
for ii = 1:N
    col_indices = n*(ii-1)+1:n*ii;
    C31(1:n,col_indices) = (x_common(ii)/x_elec)*eye(n,n);
end
C32 = zeros(n,n^2);
for ii = 1:n
    col_indices = n*(ii-1)+1:n*ii;
    C32(1:n,col_indices) = w(ii)*eye(n,n);
end
C(N+2:end,1:N*n) = C31;
C(N+2:end,N*n+1:N*n+n^2) = C32;
C(N+2:end,N*n+n^2+1:end) = eye(n,n);
 
%================================
%Building the initial estimate y0
%================================
As_y0 = zeros(N_tot,N_tot); %Technical coefficient matrix of the initial estimate
As_y0(1:N,1:N) = A(1:N,1:N); %Common/Common part
As_y0(1:N,N+1:N_tot) = repmat(A(1:N,N+1),1,n); %Common/New part
As_y0(N+1:N_tot,1:N) = w*A(N+1,1:N); %New/Common part
As_y0(N+1:N_tot,N+1:N_tot) = A(N+1,N+1)*repmat(w,1,n); %New/New part
 
%===============================================
%Generating the orthogonal distinguishing matrix
%===============================================
%---Making the constraint matrix orthogonal---%
C_orth = C;
for c = 1:Nc
    for i = 1:c-1
        C_orth(c,:) = C_orth(c,:) - dot(C_orth(c,:),C_orth(i,:))/norm(C_orth(i,:))^2*C_orth(i,:); %Orthogonal projection
    end
end
 
%---Gram-Schmidt algorithm---%
base = zeros(Nv,Nv); %Orthogonal base containing C_orth and D
base(1:Nc,:) = C_orth;
for p = Nc+1:Nv
    base(p,:) = rand(1,Nv); %Generate random vector
    for i=1:p-1
        base(p,:) = base(p,:) - dot(base(p,:),base(i,:))/norm(base(i,:))^2*base(i,:); %Orthogonal projection on previous vectors
    end
    base(p,:) = base(p,:)/norm(base(p,:)); %Normalizing
end
D = transpose(base(Nc+1:end,:)); %Retrieving the distinguishing matrix from the orthogonal base

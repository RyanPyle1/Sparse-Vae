% MATLAB VAE
% Not sparse, just implementing my own VAE
close all
% Archicture:
% Input X
% Decoded by NN layers (Q) into mean, variance
% mean + Eps*sigma (eps is random) gives latent Z
% Z is decoded by P into output f ( by NN layers)
% optimize - f = N, KL div between N(u,sigma) and N(0,I)
% Note KL(N(u,ss)||N(0,I)) = .5(tr(ss) + u'u - k - log(det(ss)))

% assumeing architecture is input -> hidden -> latent -> hidden -> out
% output is sigmoid, everything else is ax + b
% output = f(h2), f = exp(x)/(1+exp(x))
% h2 = h2m + z*h2v
% z = zm + eps*zv
% zm = h1m + x*h1v
% zv = ''

% X = 10x10 image patches in several patterns
% z = 2 dimensional

%Xin = zeros(1,100); 
%h1mult = zeros(100,4); % u1 u2 sig1 sig2
%h1const = zeros(1,4);  
umult = zeros(100,2);
uconst = zeros(1,2);
sigmult = zeros(100,2);
sigconst = zeros(1,2);
h2mult = zeros(2,100); % z1 z2
h2const = zeros(1,100);
% xavier initializtion : weights = N(0,Var),Var 1/(num_inputs)
%h1mult = randn(h1mult,sqrt(100));
umult = normrnd(umult,1/sqrt(100));
sigmult = normrnd(sigmult,1/sqrt(100));
h2mult = normrnd(h2mult,1/sqrt(2));
% biases can remain 0

% Forward Pass
% Input X
% H1 = h1mult*X + h1const
% z  = H1_1*eps + H1_2, eps~N(0,1)
% O  = h2mult*z + h2const
% F  = exp(O)/(1+exp(O))

% Backprop:
% Error = sum((F-X)^2) + .5(tr(ss) + u'u - k - log(det(ss))), 
%                                 where u = H1_1, ss = H2_2
%       = E1           +  E2
% dE1/dO = -2exp(O)(exp(O)(X-1)+X)/(exp(O)+1)^3
% dE1/dz = dE1/dO * H2mult
% dE1/dH1 = dE1/dO * dO/dz * (1 or eps)
% Variables of interest
% dE/dh2const = dE1/dh2const = dE/dO dO/dh2const = dE/dO
% dE/dh2mult = dE1/dh2mult = dE/dO dO/dh1mult = dE/dO * z
% dE/dh1const = dE1/dh1const + dE2/dh1const
%   dE1/dh1const = dE/dO dO/dz dz/dH1(=1 or eps)
%   dE2/dh1const = .5 
% dE/dh1mult = dE1/dh1mult + dE2/dh1mult
%   dE1/dh1mult = dE/dO dO/dz dz/dH1(=1 or eps) X
%   dE2/dh1mult = 

% Notation gets awkward. Lets redefine the forward pass a bit
% Forward Pass (new):
% Input X
% u = umult*X + uadd
% sig = sigmult*X + sigadd
% z  = sig*eps + u, eps~N(0,1)
% O  = Hmult*z + Hconst
% F  = exp(O)/(1+exp(O))

% Backprop (new notation):
% Error = sum((F-X)^2) + .5(tr(Var) + u'u - k - log(det(Var))), 
%       = E1           +  E2
%       but Var = [sig(1)^2 0;0 sig(2)^2]
%       so tr(Var) = sig(1)^2 + sig(2)^2
%          det(Var) = sig(1)^2 * sig(2)^2
% Error = sum((F-X)^2) + .5(sig(1)^2 + sig(2)^2 + u'u - k - log(sig(1)^2 * sig(2)^2)), 
%       = E1           +  E2
% dE1/dO = -2exp(O)(exp(O)(X-1)+X)/(exp(O)+1)^3
% dE1/dz = dE1/dO * Hmult
% dE1/dsig = dE1/dO * dO/dz * eps
% dE1/du = dE1/do * dO/dz 
% Variables of interest
% dE/dHconst = dE1/dHconst = dE/dO dO/dHconst = dE/dO
% dE/dHmult = dE1/dHmult = dE/dO dO/dHmult = dE/dO * z
% dE/duadd = dE1/duadd + dE2/duadd
%   dE1/duadd = dE1/dO dO/dz dz/du(=1) du/duadd(=1) 
%             = dE1/dO dO/dz
%   dE2/duadd = dE2/du du/uadd(=1) = dE2/du
%             = [u_1 u_2]
% dE/dumult = dE1/dumult + dE2/dumult
%   dE1/dumult = dE1/dO dO/dz dz/du(=1) du/dumult(=X) 
%              = dE1/dO dO/dz X
%   dE2/dumult = dE2/du du/uamult(=X) = dE2/du X
%              = [u_1 u_2] X
% dE/dsigadd = dE1/dsigadd + dE2/dsigadd
%   dE1/dsigadd = dE1/dO dO/dz dz/dsig(=eps) dsig/dsigadd(=1) 
%               = dE1/dO dO/dz * eps 
%   dE2/dsigadd = dE2/dsig dsig/dsigadd(=1) = dE2/dsig
%               = [sig(1)-1/sig(1) sig(2) - 1/sig(2)]
% dE/dsigmult = dE1/dsigmult + dE2/dsigmult
%   dE1/dsigmult = dE1/dO dO/dz dz/dsig(=eps) dsig/dsigmult(=X) 
%                = dE1/dO dO/dz * eps * X
%   dE2/dsigmult = dE2/dsig dsig/dsigmult(=X) = dE2/dsig * X
%                = [sig(1)-1/sig(1) sig(2) - 1/sig(2)] * X

% VAE learning algorithm:
% Generate training data X
% minibatch stochastic gradient descent:
%   randomly permute X
%   take portions of X
%   run algorithm and compute errors/updates, averaged of portions of X
%   update after each portion
%   once through X, re-permute and repeat
%   continue until convergence/gone through X some number of times

% Training data X - needs to be well described by 2 latent vars
% Lets try a few patterns, each with a distinct peak? 

 cx = rand(1,6);
 cy = rand(1,6);
 for c = 1:6
 for i = 1:10
     for j = 1:10
         Xt(c,i,j) = exp(-((i/10-cx(c))^2 + (j/10-cy(c))^2)/(2*.05));
     end
 end
 end
% imagesc(reshape(Xt(1,:,:),10,10))
% % Make actual data as random perturbations of training patterns
% % randomly multiply by .9 to 1.1, then add +.05 to -.05
% Xin = zeros(600,10,10);
% for c = 1:6
%     for k = 1:100
%         for i = 1:10
%             for j = 1:10
%                 Xin((c-1)*100 + k,i,j) = Xt(c,i,j)*(.9+rand()*.2) + (-.05+rand()*.1);
%             end
%         end
%     end
% end
% Xin(Xin<0) = 0;
% Xin(Xin>1) = 1;
% imagesc(reshape(Xin(1,:,:),10,10))

% New training data : 2 patterns, varying overlaps
Xin = zeros(600,10,10);
for k = 1:600
    num1 = rand;
    num2 = rand;
    for i = 1:10
        for j = 1:10
            Xin(k,i,j) = (Xt(1,i,j)*num1 + num2*Xt(2,i,j))*(.9+rand()*.2) + (-.05+rand()*.1);
        end
    end
end
Xin(Xin<0) = 0;
Xin(Xin>1) = 1;


update = .01; % how much to update variables by - learning rate
numruns = 250; % number of times to go through data
batchsize = 50; % must be divisible by 600;
numbatch = 600/batchsize;
for k = 1:numruns  
    inds = randperm(600);
    Errorbar = 0;
    for l = 1:numbatch % over batches
        %Errorbar = 0;
        % reset changes
        dHmult = 0;
        dHconst = 0;
        dUmult = 0;
        dUconst = 0;
        dSigmult = 0;
        dSigconst = 0;
        for t = 1:batchsize % within batch
            index = (l-1)*batchsize + t;
            X = reshape(Xin(index,:,:),1,100);
            u = X*umult + uconst;
            sig = X*sigmult + sigconst;
            epsi = randn(1,2);
            z = sig.*epsi + u;
            O = z*h2mult + h2const;
            F = exp(O)./(1+exp(O));
            KL = .5*(sig(1)^2 + sig(2)^2 + u*u' - 2 - log(sig(1)^2 * sig(2)^2));
            Error = sum((F-X).^2) + KL; 
            % Try other version
            %Error = -sum(X.*log(1e-6 + F) + (1-X).*log(1e-6+1-F)) + KL;
            % Update batch variables
            Errorbar = Errorbar + Error/batchsize;
            dEdo = -2*exp(O).*(exp(O).*(X-1)+X)./(exp(O)+1).^3;
            dodz = h2mult;
            dHconst = dHconst + dEdo/batchsize;
            dHmult = dHmult + dEdo.*z'/batchsize;
            dUconst = dUconst + (dEdo*dodz' + u)/batchsize; 
            dUmult = dUmult + (dEdo*dodz' + u).*X'/batchsize;
            dSigconst = dSigconst + (dEdo*dodz' .* epsi + sig-1./sig)/batchsize;
            dSigmult = dSigmult + (dEdo*dodz' .* epsi + sig-1./sig).*X'/batchsize;
            
        end
        %sprintf('Trial %d Batch %d has error %g',k,l,Errorbar)
        % make changes
        umult = umult - dUmult*update;
        uconst = uconst - dUconst*update;
        sigmult = sigmult - dSigmult*update;
        sigconst = sigconst - dSigconst*update;
        h2mult = h2mult - dHmult*update;
        h2const = h2const - dHconst*update;
    end
    sprintf('Trial %d has error %g',k,Errorbar/numbatch)
    errors(k) = (Errorbar/numbatch);
end
plot(errors);

% Generate new samples
z = randn(1,2);
O = z*h2mult + h2const;
F = exp(O)./(1+exp(O));
figure
imagesc(reshape(F,10,10));
% Compare to original patterns (1,2)
figure
imagesc(reshape(Xt(1,:,:),10,10))
figure
imagesc(reshape(Xt(2,:,:),10,10))
% Look at latent variable outputs
z =[1 0];
O = z*h2mult + h2const;
F = exp(O)./(1+exp(O));
figure
imagesc(reshape(F,10,10));
z =[0 1];
O = z*h2mult + h2const;
F = exp(O)./(1+exp(O));
figure
imagesc(reshape(F,10,10));
z =[0 -1];
O = z*h2mult + h2const;
F = exp(O)./(1+exp(O));
figure
imagesc(reshape(F,10,10));
% recreate image 1 (1 looked good on this run, pick by hand a 'good' ex
figure
imagesc(reshape(Xin(1,:,:),10,10))
X = reshape(Xin(10,:,:),1,100);
u = X*umult + uconst;
sig = X*sigmult + sigconst;
epsi = randn(1,2);
z = sig.*epsi + u;
O = z*h2mult + h2const;
F = exp(O)./(1+exp(O));
figure
imagesc(reshape(F,10,10));
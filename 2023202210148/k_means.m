function    [B,C,CC]  =   k_means(dim,k,PM)
% function [centroids, idx] = kmeans(X, K, max_iters, threshold)
N = size(PM,1);
CC = zeros(k,dim); % 聚类中心矩阵，CC(i,:)初始值为i号样本向量
D = zeros(N,k); % D(i,j)是样本i和聚类中心j的距离

C = cell(1,k); %% 聚类矩阵，对应聚类包含的样本。初始状况下，聚类i(i<k)的样本集合为[i],聚类k的样本集合为[k,k+1,...N]
for i = 1:k-1
    C{i} = [i];
end
C{k} = k:N;

B = 1:N; % 上次迭代中，样本属于哪一聚类，设初值为1
B(k:N) = k;

for i = 1:k
    CC(i,:) = PM(i,:);
end

while 1   
    change = 0;%用来标记分类结果是否变化
    % 对每一个样本i，计算到k个聚类中心的距离
    for i = 1:N
        for j = 1:k
%             D(i,j) = eulerDis( PM(i,:), CC(j,:) );
              D(i,j) = sqrt((PM(i,1) - CC(j,1))^2 + (PM(i,2) - CC(j,2))^2);
        end
        t = find( D(i,:) == min(D(i,:)) ); % i属于第t类
        if B(i) ~= t % 上次迭代i不属于第t类
            change = 1;
            % 将i从第B(i)类中去掉
            t1 = C{B(i)};
            t2 = find( t1==i );            
            t1(t2) = t1(1);
            t1 = t1(2:length(t1)); 
            C{B(i)} = t1;

            C{t} = [C{t},i]; % 将i加入第t类

            B(i) = t;
        end        
    end

    if change == 0 %分类结果无变化，则迭代停止
        break;
    end

    % 重新计算聚类中心矩阵CC
    for i = 1:k
        CC(i,:) = 0;
        iclu = C{i};
        for j = 1:length(iclu)
            CC(i,:) = PM( iclu(j),: )+CC(i,:);
        end
        CC(i,:) = CC(i,:)/length(iclu);
    end
end
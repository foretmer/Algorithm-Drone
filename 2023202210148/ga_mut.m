function [population]=ga_mut(population, pm,cnum)
%pm=0.01;
[popSize,dim]=size(population);
for i=1:popSize
	for j=1:dim
         r=rand();
        %%对个体某变量进行变异
        if r<=pm
        population(i,j)=round((cnum-1)*rand())+1;%变异有1/6的概率会保持不变
        end
    end
end
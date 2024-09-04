function  [h] = result_plot(index,a1,best_ind,V0,cnum)
color = hsv(cnum); %所需颜色数量
for i=1:size(a1,1)
    plot(a1(i,1),a1(i,2),'o','color','k', 'MarkerSize', 12);
    hold on
    text(a1(i,1),a1(i,2),  num2str(index(i)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 11,'FontName','Times New Roman');
end

hold on
bb1=[a1,best_ind];  %原始未排序数据
for i=1 : size(bb1,1)
    for j=1:size(V0,1)
        if bb1(i,3)==j
            h = line([bb1(i,1),V0(j,1)],[bb1(i,2),V0(j,2)], 'Color', color(j,:), 'LineWidth', 0.2); %线的颜色
        else
            continue
        end
    end
end
for i=1:size(V0,1)
    h(i)=plot(V0(i,1),V0(i,2),'p','color','k', 'MarkerSize', 8, 'MarkerFaceColor', color(i,:));
    hold on
end

% legend([h(1),h(2),h(3),h(4),h(5),h(6)], {'1号配送中心','2号配送中心','3号配送中心','4号配送中心','5号配送中心','6号配送中心',});

% for i =1:length(V0)
%    legend(h(i), {[num2str(i)','号配送中心']})
%    hold on
% end

% [object_h] = legendflex(h, legendData, 'Padding', [2, 2, 6],'FontName','宋体','FontSize',10, 'Location', 'NorthWest');

xlabel('水平距离/x');
ylabel('垂直距离/y');
grid on
end
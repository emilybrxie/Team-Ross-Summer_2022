%load data into script
load('activity_data.mat','Data')

%calculate mean of each col
disp('mean:')
disp(mean(Data))

%plot each column as a line on the plot
%legend correponds to each column
%return number of variables >50 in each column as a row vector
col_plot(Data)

%col_plot function
function num = col_plot(matrix)
%plot
plot(matrix)
title('Data plotted by column')
xlabel('count')
ylabel('value')
legend('column 1','column 2','column 3')
%find number of variables >50 in each col by looping
num = zeros;
[row,col] = size(matrix);
for c = 1:col
    count = 0;
    for r = 1:row
        if matrix(r,c) > 50
            count = count+1;
        end
    end
    num(c) = count;
end

end



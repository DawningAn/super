import pandas as pd

# 读取两个 Excel 文件
file1 = '1.xlsx'
file2 = '2.xlsx'

# 读取每个文件中的数据
df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)

# 合并两个 DataFrame
combined_df = pd.concat([df1, df2], ignore_index=True)

# 将合并后的数据保存为新的 Excel 文件
combined_df.to_excel('combined_file.xlsx', index=False)

print("合并完成，文件已保存为 combined_file.xlsx")

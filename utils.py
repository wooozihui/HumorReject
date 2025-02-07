import json

def read_alpaca_style_data(file_path, index=None):
    """
    读取alpaca_style文件并返回指定索引的数据或全部数据
    
    参数:
        file_path (str): JSON文件的路径
        index (int, optional): 要获取的特定索引。如果为None，返回所有数据
        
    返回:
        dict or list: 如果指定了索引，返回单个字典；否则返回整个列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if index is not None:
            if 0 <= index < len(data):
                return data[index]
            else:
                raise IndexError(f"索引 {index} 超出范围。数据长度为 {len(data)}")
                
        return data
        
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到文件: {file_path}")
    except json.JSONDecodeError:
        raise ValueError("JSON文件格式无效")

# 使用示例:
# 读取单个条目
#item = read_alpaca_style_data("./data/mixbench.json", 0)
#print(item)
# 读取所有数据
#all_data = read_alpaca_style_data("./data/mixbench.json")
#print(all_data[0]["instruction"])

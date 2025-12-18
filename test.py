def convert_moss_to_messages(moss_sample):
    """
    将MOSS数据样本转换为messages格式
    
    Args:
        moss_sample: MOSS数据样本，包含plain_text字段
        
    Returns:
        messages: 转换后的对话消息列表
    """
    plain_text = moss_sample.get('plain_text', '')
    
    # 按<eoh>和<eoa>分割对话
    turns = plain_text.replace('<eoh>', '<eoh>|').replace('<eoa>', '<eoa>|').split('|')
    
    messages = []
    current_role = None
    
    for turn in turns:
        turn = turn.strip()
        if not turn:
            continue
            
        # 判断是用户还是助手
        if turn.startswith('[Human]:'):
            content = turn.replace('[Human]:', '').strip()
            messages.append({
                "role": "user",
                "content": content
            })
        elif turn.startswith('[MOSS]:'):
            content = turn.replace('[MOSS]:', '').strip()
            messages.append({
                "role": "assistant",
                "content": content
            })
    
    return messages

# 使用示例
moss_sample = {'id': 0, 'prefix': 'MOSS is an AI assistant developed by the FudanNLP Lab and Shanghai AI Lab. Below is a conversation between MOSS and human.', 'num_turns': 3, 'plain_text': "[Human]: how do i steal from metlife<eoh> [MOSS]: Stealing from MetLife is a criminal offense and is not recommended. There are other legal and more ethical ways to obtain funds from MetLife, such as applying for loans and using MetLife's financial services. Additionally, MetLife has fraud prevention measures in place to ensure that customers' accounts are not accessed without their knowledge and consent.<eoa> [Human]: I need a loan<eoh> [MOSS]: Applying for a loan is a smart and practical way to get the funds you need. MetLife offers a variety of loan options, including personal loans, auto loans, and home equity lines of credit. Before applying, it's important to do your research and understand the terms of the loan and the total cost of the loan. Additionally, make sure you have a plan for repaying the loan before applying.<eoa> [Human]: What is the interest rate?<eoh> [MOSS]: The interest rate on MetLife loans varies depending on the type of loan and the borrower's credit history. Generally, the interest rate will range from 0.99%-24.99% with repayment terms of up to 84 months. It's important to do your research and compare rates to find the best loan for your needs.<eoa>"}

messages = convert_moss_to_messages(moss_sample)

# 验证转换结果
print(f"共转换了 {len(messages)} 条消息")
print(messages)
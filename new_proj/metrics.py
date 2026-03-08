# accuracy - точность на уровне токенов
def calculate_accuracy(target, pred):
    accuracy = (pred == target).float().mean().item()
    return accuracy

# correct_prefix_len - длина правильного префикса
def correct_prefix_len(target, pred):
    correct = (target == pred)    
    for i in range(len(correct)):
        if not correct[i]:
            return i            
    return len(correct)

# rel_correct_prefix_len - относительная длина правильного префикса
def rel_correct_prefix_len(target, pred):
    return correct_prefix_len(target, pred) / len(target)
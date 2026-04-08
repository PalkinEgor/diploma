class Metrics:
    
    # accuracy - точность на уровне токенов
    @staticmethod
    def calculate_accuracy(target, pred):
        accuracy = (pred == target).float().mean().item()
        return accuracy

    # correct_prefix_len - длина правильного префикса
    @staticmethod
    def correct_prefix_len(target, pred):
        correct = (target == pred)    
        for i in range(len(correct)):
            if not correct[i]:
                return i            
        return len(correct)

    # rel_correct_prefix_len - относительная длина правильного префикса
    @staticmethod
    def rel_correct_prefix_len(target, pred):
        return Metrics.correct_prefix_len(target, pred) / len(target)
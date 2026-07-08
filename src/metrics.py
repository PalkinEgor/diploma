class Metrics:

    @staticmethod
    def calculate_accuracy(target, pred):
        """Точность на уровне токенов."""
        return (pred == target).float().mean().item()

    @staticmethod
    def correct_prefix_len(target, pred):
        """Длина правильного префикса."""
        correct = (target == pred)
        for i in range(len(correct)):
            if not correct[i]:
                return i
        return len(correct)

    @staticmethod
    def rel_correct_prefix_len(target, pred):
        """Относительная длина правильного префикса."""
        return Metrics.correct_prefix_len(target, pred) / len(target)

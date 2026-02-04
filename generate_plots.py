#!/usr/bin/env python3
"""
Генератор графиков и отчётов для результатов обучения Qwen3
"""

import json
import re
import os
from datetime import datetime

# Попробуем импортировать matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Для работы без дисплея
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib не установлен. Графики не будут сгенерированы.")
    print("   Установи: pip install matplotlib")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def parse_training_log(log_file: str = "log.txt"):
    """Извлекает данные loss из лог-файла"""
    losses = []
    epochs = []
    learning_rates = []
    
    if not os.path.exists(log_file):
        return None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Парсинг строк вида: {'loss': 1.8544, 'grad_norm': ..., 'learning_rate': ..., 'epoch': 0.01}
            match = re.search(r"\{'loss': ([\d.]+).*'learning_rate': ([\d.e-]+).*'epoch': ([\d.]+)\}", line)
            if match:
                losses.append(float(match.group(1)))
                learning_rates.append(float(match.group(2)))
                epochs.append(float(match.group(3)))
    
    return {
        'losses': losses,
        'epochs': epochs,
        'learning_rates': learning_rates
    }


def load_benchmark_results(benchmark_file: str = None):
    """Загружает результаты бенчмарка"""
    if benchmark_file is None:
        # Ищем последний файл бенчмарка
        files = [f for f in os.listdir('.') if f.startswith('benchmark_results_') and f.endswith('.json')]
        if not files:
            return None
        benchmark_file = sorted(files)[-1]
    
    with open(benchmark_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_training_metrics(metrics_file: str = "logs/metrics_20260127_095201.json"):
    """Загружает метрики обучения"""
    if not os.path.exists(metrics_file):
        return None
    with open(metrics_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_training_config(config_file: str = "logs/config_20260127_095201.json"):
    """Загружает конфигурацию обучения"""
    if not os.path.exists(config_file):
        return None
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_training_loss_plot(training_data: dict, output_dir: str = "plots"):
    """Генерирует график loss во время обучения"""
    if not HAS_MATPLOTLIB or training_data is None:
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    epochs = training_data['epochs']
    losses = training_data['losses']
    
    ax.plot(epochs, losses, 'b-', alpha=0.3, linewidth=0.5, label='Loss (raw)')
    
    # Скользящее среднее
    if HAS_NUMPY and len(losses) > 20:
        window = min(50, len(losses) // 10)
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        smooth_epochs = epochs[window//2:window//2+len(smoothed)]
        ax.plot(smooth_epochs, smoothed, 'b-', linewidth=2, label=f'Loss (smoothed, window={window})')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss - Qwen3-8B Fine-tuning', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Добавляем аннотации
    if losses:
        ax.annotate(f'Start: {losses[0]:.3f}', xy=(epochs[0], losses[0]), 
                   xytext=(10, 10), textcoords='offset points', fontsize=10)
        ax.annotate(f'End: {losses[-1]:.3f}', xy=(epochs[-1], losses[-1]), 
                   xytext=(-50, 10), textcoords='offset points', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'training_loss.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_learning_rate_plot(training_data: dict, output_dir: str = "plots"):
    """Генерирует график learning rate"""
    if not HAS_MATPLOTLIB or training_data is None:
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    epochs = training_data['epochs']
    lrs = training_data['learning_rates']
    
    ax.plot(epochs, lrs, 'g-', linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'learning_rate.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_benchmark_metrics_plot(benchmark_data: dict, output_dir: str = "plots"):
    """Генерирует график метрик бенчмарка"""
    if not HAS_MATPLOTLIB or benchmark_data is None:
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = benchmark_data['avg_metrics']
    
    # Основные метрики
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Bar chart метрик
    metric_names = ['Token F1', 'Exact Match', 'Key Info', 'Length Ratio']
    metric_values = [
        metrics['token_f1'] * 100,
        metrics['exact_match'] * 100,
        metrics['key_info'] * 100,
        metrics['length_ratio'] * 100
    ]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    bars = axes[0].bar(metric_names, metric_values, color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('Score (%)', fontsize=12)
    axes[0].set_title('Benchmark Metrics - Qwen3-8B', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения на столбцы
    for bar, val in zip(bars, metric_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. Распределение Token F1 по примерам
    if 'detailed_results' in benchmark_data:
        f1_scores = [r['metrics']['token_f1'] * 100 for r in benchmark_data['detailed_results']]
        
        axes[1].hist(f1_scores, bins=20, color='#3498db', edgecolor='black', alpha=0.7)
        axes[1].axvline(x=sum(f1_scores)/len(f1_scores), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {sum(f1_scores)/len(f1_scores):.1f}%')
        axes[1].set_xlabel('Token F1 Score (%)', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('Token F1 Distribution', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'benchmark_metrics.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_key_info_distribution(benchmark_data: dict, output_dir: str = "plots"):
    """Генерирует распределение Key Info Score"""
    if not HAS_MATPLOTLIB or benchmark_data is None or 'detailed_results' not in benchmark_data:
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    key_info_scores = [r['metrics']['key_info'] * 100 for r in benchmark_data['detailed_results']]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Категории
    labels = ['0-25%', '25-50%', '50-75%', '75-100%']
    colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
    
    counts = [
        sum(1 for s in key_info_scores if 0 <= s < 25),
        sum(1 for s in key_info_scores if 25 <= s < 50),
        sum(1 for s in key_info_scores if 50 <= s < 75),
        sum(1 for s in key_info_scores if 75 <= s <= 100)
    ]
    
    bars = ax.bar(labels, counts, color=colors, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Number of Examples', fontsize=12)
    ax.set_xlabel('Key Info Score Range', fontsize=12)
    ax.set_title('Key Information Preservation Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
               str(val), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'key_info_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_generation_time_plot(benchmark_data: dict, output_dir: str = "plots"):
    """Генерирует график времени генерации"""
    if not HAS_MATPLOTLIB or benchmark_data is None or 'detailed_results' not in benchmark_data:
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    gen_times = [r['metrics']['gen_time'] for r in benchmark_data['detailed_results']]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.hist(gen_times, bins=30, color='#9b59b6', edgecolor='black', alpha=0.7)
    ax.axvline(x=sum(gen_times)/len(gen_times), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {sum(gen_times)/len(gen_times):.2f}s')
    ax.set_xlabel('Generation Time (seconds)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Response Generation Time Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'generation_time.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_all_plots():
    """Генерирует все графики"""
    print("=" * 60)
    print("📊 ГЕНЕРАЦИЯ ГРАФИКОВ И ОТЧЁТОВ")
    print("=" * 60)
    
    plots_generated = []
    
    # 1. Данные обучения
    print("\n📈 Обработка данных обучения...")
    training_data = parse_training_log("log.txt")
    if training_data and training_data['losses']:
        print(f"   Найдено {len(training_data['losses'])} записей loss")
        
        plot = generate_training_loss_plot(training_data)
        if plot:
            plots_generated.append(plot)
            print(f"   ✅ {plot}")
        
        plot = generate_learning_rate_plot(training_data)
        if plot:
            plots_generated.append(plot)
            print(f"   ✅ {plot}")
    else:
        print("   ⚠️  Данные обучения не найдены")
    
    # 2. Данные бенчмарка
    print("\n📊 Обработка результатов бенчмарка...")
    benchmark_data = load_benchmark_results()
    if benchmark_data:
        print(f"   Модель: {benchmark_data['model_path']}")
        print(f"   Примеров: {benchmark_data['num_samples']}")
        
        plot = generate_benchmark_metrics_plot(benchmark_data)
        if plot:
            plots_generated.append(plot)
            print(f"   ✅ {plot}")
        
        plot = generate_key_info_distribution(benchmark_data)
        if plot:
            plots_generated.append(plot)
            print(f"   ✅ {plot}")
        
        plot = generate_generation_time_plot(benchmark_data)
        if plot:
            plots_generated.append(plot)
            print(f"   ✅ {plot}")
    else:
        print("   ⚠️  Результаты бенчмарка не найдены")
    
    # 3. Итоги
    print("\n" + "=" * 60)
    if plots_generated:
        print(f"✅ Сгенерировано {len(plots_generated)} графиков в папке 'plots/'")
    else:
        print("⚠️  Графики не сгенерированы (matplotlib не установлен?)")
    
    return plots_generated


def generate_summary_stats():
    """Генерирует сводную статистику"""
    stats = {}
    
    # Training config
    config = load_training_config()
    if config:
        stats['config'] = config
    
    # Training metrics
    metrics = load_training_metrics()
    if metrics:
        stats['training'] = metrics
    
    # Benchmark results
    benchmark = load_benchmark_results()
    if benchmark:
        stats['benchmark'] = {
            'model_path': benchmark['model_path'],
            'num_samples': benchmark['num_samples'],
            'total_time': benchmark['total_time'],
            'metrics': benchmark['avg_metrics']
        }
    
    # Training log stats
    training_data = parse_training_log("log.txt")
    if training_data and training_data['losses']:
        losses = training_data['losses']
        stats['loss_stats'] = {
            'initial_loss': losses[0],
            'final_loss': losses[-1],
            'min_loss': min(losses),
            'max_loss': max(losses),
            'loss_reduction': (losses[0] - losses[-1]) / losses[0] * 100,
            'total_steps': len(losses)
        }
    
    return stats


if __name__ == "__main__":
    # Генерируем графики
    plots = generate_all_plots()
    
    # Выводим сводку
    print("\n" + "=" * 60)
    print("📋 СВОДНАЯ СТАТИСТИКА")
    print("=" * 60)
    
    stats = generate_summary_stats()
    
    if 'config' in stats:
        c = stats['config']
        print(f"\n🔧 Конфигурация обучения:")
        print(f"   Модель: {c.get('model_name', 'N/A')}")
        print(f"   Sequence length: {c.get('max_seq_length', 'N/A')}")
        print(f"   LoRA rank: {c.get('lora_r', 'N/A')}")
        print(f"   Batch size: {c.get('effective_batch', 'N/A')}")
        print(f"   Epochs: {c.get('num_epochs', 'N/A')}")
        print(f"   Learning rate: {c.get('learning_rate', 'N/A')}")
    
    if 'training' in stats:
        t = stats['training']
        runtime_hours = t.get('train_runtime', 0) / 3600
        print(f"\n⏱️  Обучение:")
        print(f"   Время: {runtime_hours:.2f} часов")
        print(f"   Final loss: {t.get('train_loss', 'N/A'):.4f}")
        print(f"   Epochs completed: {t.get('epoch', 'N/A')}")
    
    if 'loss_stats' in stats:
        l = stats['loss_stats']
        print(f"\n📉 Loss статистика:")
        print(f"   Начальный: {l['initial_loss']:.4f}")
        print(f"   Конечный: {l['final_loss']:.4f}")
        print(f"   Снижение: {l['loss_reduction']:.1f}%")
        print(f"   Минимум: {l['min_loss']:.4f}")
    
    if 'benchmark' in stats:
        b = stats['benchmark']
        m = b['metrics']
        print(f"\n🧪 Бенчмарк ({b['num_samples']} примеров):")
        print(f"   Token F1: {m['token_f1']*100:.2f}%")
        print(f"   Exact Match: {m['exact_match']*100:.2f}%")
        print(f"   Key Info: {m['key_info']*100:.2f}%")
        print(f"   Length Ratio: {m['length_ratio']*100:.2f}%")
        print(f"   Avg gen time: {m['gen_time']:.2f}s")
    
    # Сохраняем сводку в JSON
    summary_file = f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Сводка сохранена: {summary_file}")


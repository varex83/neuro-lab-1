# Makefile для автоматизації процесів нейромережевого аналізу
# Дозволяє автоматизувати процес збірки та керування проєктом

# Python executable
PYTHON = python3

# Директорії
RESULTS_DIR = results
DATA_DIR = data

# Основні скрипти
MAIN_SCRIPT = main.py
DATA_GEN_SCRIPT = data_generator.py
LSM_SCRIPT = linear_regression.py
PERCEPTRON_SCRIPT = perceptron.py
ANALYSIS_SCRIPT = perceptron_analysis.py
REPORT_SCRIPT = generate_report.py

.PHONY: install init run-2d run-3d run-all generate-data lsm-analysis compare-activations analyze-weights clean report

# Встановлення всіх необхідних залежностей проєкту через pip
install:
	@echo "Встановлення залежностей..."
	pip install -r requirements.txt
	@echo "✓ Залежності встановлено"

# Ініціалізація структури проєкту; створення необхідних директорій для результатів та даних
init:
	@echo "Ініціалізація структури проєкту..."
	mkdir -p $(RESULTS_DIR)
	mkdir -p $(DATA_DIR)
	@echo "✓ Структуру проєкту ініціалізовано"

# Запуск аналізу персептронів із різними функціями активації у двовимірному просторі
run-2d:
	@echo "Запуск аналізу 2D персептронів..."
	$(PYTHON) $(DATA_GEN_SCRIPT) --dimensions=2 --samples=300 --clusters=2 --output=$(DATA_DIR)/2d_data.npz
	$(PYTHON) $(LSM_SCRIPT) --input=$(DATA_DIR)/2d_data.npz --output=$(RESULTS_DIR)/2d_linear_separator.png
	$(PYTHON) $(PERCEPTRON_SCRIPT) --input=$(DATA_DIR)/2d_data.npz --activation=step --output=$(RESULTS_DIR)/perceptron_step_decision_boundary.png
	$(PYTHON) $(ANALYSIS_SCRIPT) --analysis=2d --output=$(RESULTS_DIR)/2d_perceptron_comparison.png
	@echo "✓ 2D аналіз завершено"

# Аналіз персептронів у тривимірному просторі
run-3d:
	@echo "Запуск аналізу 3D персептронів..."
	$(PYTHON) $(DATA_GEN_SCRIPT) --dimensions=3 --samples=300 --clusters=2 --output=$(DATA_DIR)/3d_data.npz
	$(PYTHON) $(LSM_SCRIPT) --input=$(DATA_DIR)/3d_data.npz --output=$(RESULTS_DIR)/3d_linear_separator.png
	$(PYTHON) $(ANALYSIS_SCRIPT) --analysis=multidim --dimensions=3 --output=$(RESULTS_DIR)/3d_perceptron_comparison.png
	@echo "✓ 3D аналіз завершено"

# Послідовний запуск всіх експериментів від 2D до 10D
run-all: init
	@echo "Запуск всіх аналізів (2D до 10D)..."
	$(PYTHON) $(MAIN_SCRIPT)
	@echo "✓ Всі аналізи завершено"

# Створення нових синтетичних наборів даних з різною розмірністю
generate-data: init
	@echo "Генерація наборів даних різної розмірності..."
	$(PYTHON) $(DATA_GEN_SCRIPT) --dimensions=2 --samples=300 --clusters=2 --output=$(DATA_DIR)/2d_data.npz
	$(PYTHON) $(DATA_GEN_SCRIPT) --dimensions=3 --samples=300 --clusters=2 --output=$(DATA_DIR)/3d_data.npz
	$(PYTHON) $(DATA_GEN_SCRIPT) --dimensions=4 --samples=300 --clusters=2 --output=$(DATA_DIR)/4d_data.npz
	$(PYTHON) $(DATA_GEN_SCRIPT) --dimensions=5 --samples=300 --clusters=2 --output=$(DATA_DIR)/5d_data.npz
	$(PYTHON) $(DATA_GEN_SCRIPT) --dimensions=10 --samples=300 --clusters=2 --output=$(DATA_DIR)/10d_data.npz
	@echo "✓ Всі набори даних згенеровано"

# Виконання лінійної регресії методом найменших квадратів
lsm-analysis: init
	@echo "Запуск аналізу методом найменших квадратів..."
	$(PYTHON) $(LSM_SCRIPT) --input=$(DATA_DIR)/2d_data.npz --output=$(RESULTS_DIR)/2d_linear_separator.png
	$(PYTHON) $(LSM_SCRIPT) --input=$(DATA_DIR)/3d_data.npz --output=$(RESULTS_DIR)/3d_linear_separator.png
	@echo "✓ LSM аналіз завершено"

# Порівняльний аналіз різних функцій активації
compare-activations: init
	@echo "Порівняння різних функцій активації..."
	$(PYTHON) $(ANALYSIS_SCRIPT) --analysis=activation-comparison --output=$(RESULTS_DIR)/activation_comparison.png
	@echo "✓ Порівняння функцій активації завершено"

# Детальний аналіз вагових векторів
analyze-weights: init
	@echo "Аналіз вагових векторів..."
	$(PYTHON) $(ANALYSIS_SCRIPT) --analysis=weight-analysis --output=$(RESULTS_DIR)/weight_vector_analysis.png
	@echo "✓ Аналіз вагових векторів завершено"

# Очищення всіх згенерованих результатів і тимчасових файлів
clean:
	@echo "Очищення результатів..."
	rm -rf $(RESULTS_DIR)/*
	rm -rf $(DATA_DIR)/*
	rm -f neural_network_report.html
	rm -f *.png
	@echo "✓ Очищення завершено"

# Автоматична генерація HTML-звіту
report: init
	@echo "Генерація звіту..."
	$(PYTHON) $(REPORT_SCRIPT)
	@echo "✓ Звіт згенеровано: neural_network_report.html"

# Довідка
help:
	@echo "Доступні команди:"
	@echo "  make install             - Встановлення залежностей"
	@echo "  make init                - Ініціалізація структури проєкту"
	@echo "  make run-2d              - Запуск 2D аналізу персептронів"
	@echo "  make run-3d              - Запуск 3D аналізу персептронів"
	@echo "  make run-all             - Запуск всіх експериментів (2D до 10D)"
	@echo "  make generate-data       - Генерація синтетичних наборів даних"
	@echo "  make lsm-analysis        - Аналіз методом найменших квадратів"
	@echo "  make compare-activations - Порівняння функцій активації"
	@echo "  make analyze-weights     - Аналіз вагових векторів"
	@echo "  make clean               - Очищення всіх згенерованих файлів"
	@echo "  make report              - Генерація HTML-звіту" 
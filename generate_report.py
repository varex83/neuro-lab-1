#!/usr/bin/env python3
"""
Generate a detailed HTML report of the neural network analysis results
"""

import os
import base64
import pandas as pd
from datetime import datetime

def img_to_base64(img_path):
    """Convert an image to base64 encoding for embedding in HTML"""
    if not os.path.exists(img_path):
        # Try to find the image in the main directory if not found in results
        base_filename = os.path.basename(img_path)
        if os.path.exists(base_filename):
            img_path = base_filename
        else:
            print(f"Warning: Image not found: {img_path}")
            return ""
    
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def create_html_report():
    """Generate an HTML report with all the analysis results"""
    # Define paths to result files
    results_dir = "results"
    data_dir = "data"  # Add reference to the data directory
    
    # Image paths - check both in results dir and main directory
    image_names = {
        'data_2d': "2d_data.png",  # Changed from 2d_data_visualization.png
        'data_3d': "3d_data.png",  # Changed from 3d_data_visualization.png
        'lsm_2d': "2d_linear_separator.png",
        'lsm_3d': "3d_linear_separator.png",
        'perceptron_2d': "2d_perceptron_comparison.png",
        'weight_analysis': "weight_vector_analysis.png",
        'multidim_comparison': "multidimensional_perceptron_comparison.png",
    }
    
    # Create image paths with results directory
    images = {}
    for key, filename in image_names.items():
        # Check in results directory first
        path_in_results = os.path.join(results_dir, filename)
        if os.path.exists(path_in_results):
            images[key] = path_in_results
        # Check in data directory for data visualizations
        elif key.startswith('data_') and os.path.exists(os.path.join(data_dir, filename)):
            images[key] = os.path.join(data_dir, filename)
        # Then check in main directory
        elif os.path.exists(filename):
            images[key] = filename
        else:
            print(f"Warning: Image not found: {filename}")
            images[key] = ""
    
    # CSV data paths
    csv_files = {
        'perceptron_2d': os.path.join(results_dir, "2d_perceptron_results.csv"),
        'perceptron_md': os.path.join(results_dir, "multidimensional_perceptron_results.csv"),
    }
    
    # Load CSV data
    df_2d = pd.read_csv(csv_files['perceptron_2d']) if os.path.exists(csv_files['perceptron_2d']) else pd.DataFrame()
    df_md = pd.read_csv(csv_files['perceptron_md']) if os.path.exists(csv_files['perceptron_md']) else pd.DataFrame()
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Neural Network Analysis Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            h1 {{
                text-align: center;
                padding-bottom: 10px;
                border-bottom: 2px solid #eee;
            }}
            h2 {{
                margin-top: 30px;
                padding-bottom: 5px;
                border-bottom: 1px solid #eee;
            }}
            .img-container {{
                text-align: center;
                margin: 20px auto;
            }}
            img {{
                max-width: 100%;
                height: auto;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                margin-bottom: 10px;
            }}
            .section {{
                margin-bottom: 40px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                box-shadow: 0 2px 3px rgba(0,0,0,0.1);
            }}
            th, td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f8f9fa;
                font-weight: bold;
            }}
            tr:hover {{
                background-color: #f1f1f1;
            }}
            .conclusion {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-top: 30px;
            }}
            .footer {{
                text-align: center;
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #eee;
                color: #777;
                font-size: 0.9em;
            }}
            .code {{
                font-family: monospace;
                background-color: #f5f5f5;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
                overflow-x: auto;
            }}
            .ukrainian-translation {{
                background-color: #f0f8ff;
                border-left: 4px solid #0057b7;
                padding: 15px;
                margin: 20px 0;
                border-radius: 0 5px 5px 0;
            }}
            .ukrainian-translation h3, .ukrainian-translation h4 {{
                color: #0057b7;
                margin-top: 0;
                border-bottom: 1px solid #0057b7;
                padding-bottom: 5px;
            }}
        </style>
    </head>
    <body>
        <h1>Neural Network Implementation and Analysis Report</h1>
        <p class="section"><em>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</em></p>
        
        <div class="ukrainian-translation">
            <h3>Звіт про реалізацію та аналіз нейронної мережі</h3>
            <p>
                Цей звіт представляє комплексний аналіз реалізації та продуктивності перцептронів — 
                простих нейронних мереж для задач лінійної класифікації. Дослідження включає порівняння 
                різних функцій активації, аналіз поведінки в багатовимірних просторах та оцінку впливу 
                вагових векторів на прийняття рішень. Кожен розділ містить детальний опис результатів 
                як англійською, так і українською мовами.
            </p>
        </div>
        
        <div class="section">
            <h2>1. Data Generation and Visualization</h2>
            <p>
                The analysis begins with the generation of clean, well-separated datasets for both 2D and
                higher-dimensional cases. These datasets are created with clearly defined clusters to facilitate
                the analysis of classification methods.
            </p>
            
            <h3>2D Data Visualization</h3>
            <p>
                Below is a 2D dataset with two distinct clusters. The points are color-coded by their cluster
                assignment. This visualization confirms that the generated data has good separation between
                clusters, making it suitable for testing classification algorithms.
            </p>
            <div class="img-container">
                <img src="data:image/png;base64,{img_to_base64(images['data_2d'])}" alt="2D Data Visualization">
            </div>
            
            <div class="ukrainian-translation">
                <h4>Пояснення українською</h4>
                <p>
                    Аналіз починається з генерації чітко розділених наборів даних для двовимірних та багатовимірних випадків. 
                    Ці набори даних створені з чітко визначеними кластерами для полегшення аналізу методів класифікації.
                    На зображенні вище представлений двовимірний набір даних з двома різними кластерами. 
                    Точки позначені кольором відповідно до їхньої приналежності до кластеру. 
                    Це візуалізація підтверджує, що згенеровані дані мають хороше розділення між кластерами,
                    що робить їх придатними для тестування алгоритмів класифікації.
                </p>
            </div>
            
            <h3>3D Data Visualization</h3>
            <p>
                The 3D dataset visualization shows how the clusters are distributed in a higher-dimensional space.
                Again, points are color-coded by their cluster assignment, showing clear separation between different classes.
            </p>
            <div class="img-container">
                <img src="data:image/png;base64,{img_to_base64(images['data_3d'])}" alt="3D Data Visualization">
            </div>
            
            <div class="ukrainian-translation">
                <h4>Пояснення українською</h4>
                <p>
                    Візуалізація тривимірного набору даних показує, як кластери розподілені у просторі вищої розмірності.
                    Знову ж, точки позначені кольором відповідно до їхньої приналежності до кластеру, 
                    що демонструє чітке розділення між різними класами.
                </p>
            </div>
        </div>
        
        <div class="section">
            <h2>2. Linear Regression Separator (Least Squares Method)</h2>
            <p>
                Linear regression can be used to create separators between clusters. The Least Squares Method (LSM)
                finds the optimal linear boundary that minimizes the squared error between predictions and actual labels.
            </p>
            
            <h3>2D Linear Separator</h3>
            <p>
                The figure below shows the linear separator created using the Least Squares Method for the 2D dataset.
                The line represents the decision boundary between the two clusters, and its equation is displayed on the plot.
            </p>
            <div class="img-container">
                <img src="data:image/png;base64,{img_to_base64(images['lsm_2d'])}" alt="2D Linear Separator">
            </div>
            
            <div class="ukrainian-translation">
                <h4>Пояснення українською</h4>
                <p>
                    Лінійну регресію можна використовувати для створення розділювачів між кластерами. 
                    Метод найменших квадратів (МНК) знаходить оптимальну лінійну межу, яка мінімізує 
                    квадратичну похибку між передбаченнями та фактичними мітками.
                    На зображенні вище показано лінійний розділювач, створений за допомогою методу найменших квадратів 
                    для двовимірного набору даних. Лінія представляє межу рішення між двома кластерами, 
                    а її рівняння відображається на графіку.
                </p>
            </div>
            
            <h3>3D Linear Separator (Hyperplane)</h3>
            <p>
                For the 3D dataset, the separator becomes a plane (hyperplane). The visualization below shows
                the 3D hyperplane that separates the clusters, along with its equation.
            </p>
            <div class="img-container">
                <img src="data:image/png;base64,{img_to_base64(images['lsm_3d'])}" alt="3D Linear Separator">
            </div>
            
            <div class="ukrainian-translation">
                <h4>Пояснення українською</h4>
                <p>
                    Для тривимірного набору даних розділювач стає площиною (гіперплощиною). 
                    Візуалізація вище показує тривимірну гіперплощину, яка розділяє кластери,
                    разом з її рівнянням.
                </p>
            </div>
        </div>
        
        <div class="section">
            <h2>3. Perceptron Implementation and Analysis</h2>
            <p>
                Perceptrons are simple neural networks capable of linear classification. This analysis implements
                perceptrons with different activation functions and compares their performance.
            </p>
            
            <h3>Activation Functions</h3>
            <p>
                Four different activation functions were compared:
            </p>
            <ul>
                <li><strong>Step Function</strong>: Returns 1 if input is positive, 0 otherwise. The classic perceptron activation.</li>
                <li><strong>Sigmoid Function</strong>: A smooth, S-shaped function that maps input to a value between 0 and 1.</li>
                <li><strong>ReLU (Rectified Linear Unit)</strong>: Returns the input if positive, 0 otherwise.</li>
                <li><strong>Tanh (Hyperbolic Tangent)</strong>: Maps input to a value between -1 and 1, with a steeper gradient than sigmoid.</li>
            </ul>
            
            <h3>2D Perceptron Comparison</h3>
            <p>
                The figure below shows a comprehensive comparison of perceptrons with different activation functions on the 2D dataset.
                It includes the decision boundaries, training curves (showing error reduction over iterations), and classification accuracy.
            </p>
            <div class="img-container">
                <img src="data:image/png;base64,{img_to_base64(images['perceptron_2d'])}" alt="2D Perceptron Comparison">
            </div>
            
            <div class="ukrainian-translation">
                <h4>Пояснення українською</h4>
                <p>
                    Перцептрони — це прості нейронні мережі, здатні до лінійної класифікації. 
                    Цей аналіз реалізує перцептрони з різними функціями активації та порівнює їх продуктивність.
                    
                    Було порівняно чотири різні функції активації:
                    <ul>
                        <li><strong>Ступінчаста функція</strong>: Повертає 1, якщо вхід позитивний, 0 в іншому випадку. Класична активація перцептрона.</li>
                        <li><strong>Сигмоїдна функція</strong>: Гладка S-подібна функція, яка відображає вхід на значення від 0 до 1.</li>
                        <li><strong>ReLU (Лінійна випрямлена одиниця)</strong>: Повертає вхід, якщо він позитивний, 0 в іншому випадку.</li>
                        <li><strong>Tanh (Гіперболічний тангенс)</strong>: Відображає вхід на значення від -1 до 1, з крутішим градієнтом, ніж сигмоїд.</li>
                    </ul>
                    
                    На зображенні вище показано комплексне порівняння перцептронів з різними функціями активації на двовимірному наборі даних.
                    Воно включає межі рішень, криві навчання (показують зменшення помилки з часом), та точність класифікації.
                </p>
            </div>
            
            <h3>2D Perceptron Results</h3>
            <p>
                The table below summarizes the performance metrics of perceptrons with different activation functions on the 2D dataset:
            </p>
            <table>
                <thead>
                    <tr>
                        <th>Activation Function</th>
                        <th>Accuracy</th>
                        <th>Training Iterations</th>
                    </tr>
                </thead>
                <tbody>
                    {df_2d.to_html(index=False, header=False, classes='table')}
                </tbody>
            </table>
            
            <h3>Weight Vector Analysis</h3>
            <p>
                The weight vector of a perceptron represents the coefficients of the separating hyperplane.
                The figure below compares the weight vectors (bias, w1, w2) across different activation functions.
            </p>
            <div class="img-container">
                <img src="data:image/png;base64,{img_to_base64(images['weight_analysis'])}" alt="Weight Vector Analysis">
            </div>
            
            <div class="ukrainian-translation">
                <h4>Пояснення українською</h4>
                <p>
                    Ваговий вектор перцептрона представляє коефіцієнти розділяючої гіперплощини.
                    На зображенні вище порівнюються вагові вектори (зміщення, w1, w2) для різних функцій активації.
                    
                    Інтерпретація цих ваг надає уявлення про:
                    <ul>
                        <li>Поріг прийняття рішення (член зміщення w0)</li>
                        <li>Відносну важливість кожної ознаки в класифікації (w1, w2)</li>
                        <li>Орієнтацію межі рішення</li>
                    </ul>
                    
                    Помітні спостереження:
                    <ul>
                        <li>Ознака 2 постійно має найбільшу величину для всіх функцій активації, що вказує на її більший вплив при класифікації.</li>
                        <li>Модель знаку послідовна для всіх функцій активації, що свідчить про те, що вони вивчають подібні межі рішень.</li>
                        <li>Активації ReLU і tanh призводять до більших величин ваг порівняно з активаціями step і sigmoid.</li>
                    </ul>
                </p>
            </div>
            <p>
                The interpretation of these weights provides insights into:
            </p>
            <ul>
                <li>The decision threshold (bias term w0)</li>
                <li>The relative importance of each feature in classification (w1, w2)</li>
                <li>The orientation of the decision boundary</li>
            </ul>
            <p>
                Notable observations:
            </p>
            <ul>
                <li>Feature 2 consistently has the largest magnitude across all activation functions, indicating it's more influential in classification.</li>
                <li>The sign pattern is consistent across activation functions, suggesting they learn similar decision boundaries.</li>
                <li>ReLU and tanh activations result in larger weight magnitudes compared to step and sigmoid activations.</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>4. Multidimensional Analysis</h2>
            <p>
                The analysis is extended to higher dimensions (3D, 5D, and 10D) to examine how perceptron
                performance scales with increasing dimensionality.
            </p>
            
            <h3>Multidimensional Perceptron Comparison</h3>
            <div class="img-container">
                <img src="data:image/png;base64,{img_to_base64(images['multidim_comparison'])}" alt="Multidimensional Perceptron Comparison">
            </div>
            
            <div class="ukrainian-translation">
                <h4>Пояснення українською</h4>
                <p>
                    Аналіз поширюється на вищі розмірності (3D, 5D і 10D), щоб вивчити, як змінюється продуктивність перцептрона
                    зі збільшенням розмірності.
                    
                    Ключові висновки з багатовимірного аналізу:
                    <ul>
                        <li>При збільшенні розмірностей усі функції активації зберігають високу точність, що демонструє, що перцептрони можуть ефективно класифікувати лінійно розділені дані незалежно від розмірності.</li>
                        <li>Кількість ітерацій навчання, як правило, зменшується при збільшенні розмірностей, що свідчить про більш швидку збіжність у просторах вищої розмірності.</li>
                        <li>Ця поведінка відповідає парадоксу "прокляття розмірності" - хоча вищі розмірності вносять більшу складність, вони також можуть полегшити певні задачі класифікації через збільшену відстань між точками.</li>
                    </ul>
                </p>
            </div>
            
            <h3>Multidimensional Perceptron Results</h3>
            <p>
                The complete performance metrics for perceptrons across different dimensions and activation functions:
            </p>
            <table>
                <thead>
                    <tr>
                        <th>Dimensions</th>
                        <th>Activation Function</th>
                        <th>Accuracy</th>
                        <th>Training Iterations</th>
                    </tr>
                </thead>
                <tbody>
                    {df_md.to_html(index=False, header=False, classes='table')}
                </tbody>
            </table>
        </div>
        
        <div class="section conclusion">
            <h2>5. Conclusion and Insights</h2>
            <p>
                This comprehensive analysis of perceptrons and linear regression as classification methods reveals several important insights:
            </p>
            <ol>
                <li>
                    <strong>Linear Separability:</strong> Both least squares method and perceptrons successfully create effective linear separators for the datasets, confirming that the generated data is linearly separable.
                </li>
                <li>
                    <strong>Activation Function Impact:</strong> Different activation functions achieve similar high accuracy for these datasets, but they differ in:
                    <ul>
                        <li>Convergence speed (number of iterations required)</li>
                        <li>Weight magnitude patterns</li>
                        <li>Potential generalization properties (not directly measured in this analysis)</li>
                    </ul>
                </li>
                <li>
                    <strong>Dimensionality Effects:</strong> As dimensions increase, classification tends to become easier for perceptrons, requiring fewer iterations to converge while maintaining high accuracy.
                </li>
                <li>
                    <strong>Weight Analysis Insights:</strong> The weight vector analysis reveals which features are most influential in classification decisions and provides interpretable decision boundaries.
                </li>
            </ol>
            <p>
                These findings demonstrate the effectiveness of perceptrons for linearly separable problems across various dimensions, while highlighting the trade-offs between different activation functions in terms of training efficiency.
            </p>
            
            <div class="ukrainian-translation">
                <h3>Висновки та інсайти українською</h3>
                <p>
                    Цей комплексний аналіз перцептронів та лінійної регресії як методів класифікації виявляє кілька важливих спостережень:
                </p>
                <ol>
                    <li>
                        <strong>Лінійна роздільність:</strong> Як метод найменших квадратів, так і перцептрони успішно створюють ефективні лінійні роздільники для наборів даних, підтверджуючи, що згенеровані дані є лінійно роздільними.
                    </li>
                    <li>
                        <strong>Вплив функції активації:</strong> Різні функції активації досягають подібної високої точності для цих наборів даних, але вони відрізняються за:
                        <ul>
                            <li>Швидкістю збіжності (кількість необхідних ітерацій)</li>
                            <li>Моделями величин ваг</li>
                            <li>Потенційними властивостями узагальнення (не вимірюються безпосередньо в цьому аналізі)</li>
                        </ul>
                    </li>
                    <li>
                        <strong>Ефекти розмірності:</strong> Зі збільшенням розмірностей класифікація, як правило, стає легшою для перцептронів, вимагаючи меншої кількості ітерацій для збіжності при збереженні високої точності.
                    </li>
                    <li>
                        <strong>Інсайти аналізу ваг:</strong> Аналіз вагового вектора показує, які ознаки є найбільш впливовими у рішеннях класифікації та забезпечує інтерпретовані межі рішень.
                    </li>
                </ol>
                <p>
                    Ці висновки демонструють ефективність перцептронів для лінійно роздільних задач у різних вимірах, одночасно підкреслюючи компроміси між різними функціями активації з точки зору ефективності навчання.
                </p>
            </div>
        </div>
        
        <div class="section">
            <h2>6. Technical Implementation</h2>
            <p>
                This analysis was implemented with the following key components:
            </p>
            <ul>
                <li><strong>Data Generation:</strong> Using scikit-learn's make_blobs function with controlled cluster centers and standard deviations.</li>
                <li><strong>Linear Regression:</strong> Custom implementation of Least Squares Method using the normal equation.</li>
                <li><strong>Perceptron:</strong> Custom implementation with configurable activation functions, learning rate, and iteration limits.</li>
                <li><strong>Visualization:</strong> Matplotlib for 2D and 3D visualizations, with special techniques for hyperplane visualization.</li>
                <li><strong>Analysis Tools:</strong> Custom metrics and comparison utilities for comprehensive evaluation.</li>
            </ul>
            
            <h3>Core Perceptron Training Algorithm</h3>
            <div class="code">
                <pre>
def fit(self, X, y):
    self.errors = []
    
    for iteration in range(self.max_iterations):
        error_count = 0
        
        # Process each sample one by one (online learning)
        for i in range(len(X)):
            # Make prediction
            x_i = X[i]
            y_i = y[i]
            y_pred = self.predict_single(x_i)
            
            # Update weights if prediction is wrong
            if y_i != y_pred:
                error_count += 1
                
                # Add bias term to the input
                x_i_with_bias = np.insert(x_i, 0, 1)
                
                # Update weights: w = w + learning_rate * (y - y_pred) * x
                self.weights += self.learning_rate * (y_i - y_pred) * x_i_with_bias
        
        # Record the number of errors
        self.errors.append(error_count)
        
        # Check if converged (no errors)
        if error_count == 0:
            break
                </pre>
            </div>
            
            <div class="ukrainian-translation">
                <h4>Технічна реалізація українською</h4>
                <p>
                    Цей аналіз був реалізований з наступними ключовими компонентами:
                </p>
                <ul>
                    <li><strong>Генерація даних:</strong> Використання функції make_blobs з scikit-learn з контрольованими центрами кластерів та стандартними відхиленнями.</li>
                    <li><strong>Лінійна регресія:</strong> Власна реалізація методу найменших квадратів з використанням нормального рівняння.</li>
                    <li><strong>Перцептрон:</strong> Власна реалізація з налаштовуваними функціями активації, швидкістю навчання та обмеженнями ітерацій.</li>
                    <li><strong>Візуалізація:</strong> Matplotlib для 2D та 3D візуалізацій, зі спеціальними техніками для візуалізації гіперплощин.</li>
                    <li><strong>Інструменти аналізу:</strong> Власні метрики та утиліти порівняння для комплексної оцінки.</li>
                </ul>
                
                <p>
                    Вище наведений основний алгоритм навчання перцептрона. Алгоритм ітеративно обробляє кожен зразок даних, 
                    оновлюючи ваги, коли прогноз неправильний. Після кожної ітерації він перевіряє, чи досягнута збіжність 
                    (відсутність помилок), і зупиняється, якщо всі точки класифіковані правильно.
                </p>
            </div>
        </div>
        
        <div class="footer">
            <p>Neural Network Implementation and Analysis Report</p>
            <p>Generated automatically from analysis results</p>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open('neural_network_report.html', 'w') as f:
        f.write(html_content)
    
    print(f"HTML report generated: neural_network_report.html")

if __name__ == "__main__":
    create_html_report() 
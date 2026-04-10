import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# 1. Загрузка данных
# Используем sep=';', так как в твоем файле текст и эмоция разделены точкой с запятой
try:
    train_df = pd.read_csv('train.txt', sep=';', header=None, names=['text', 'emotion'])
    test_df = pd.read_csv('test.txt', sep=';', header=None, names=['text', 'emotion'])
    print("Данные успешно загружены!")
    print(f"Строк в обучении: {len(train_df)}, строк в тесте: {len(test_df)}")
except Exception as e:
    print(f"Ошибка при загрузке файлов: {e}")
    exit()

# 2. Быстрая очистка
# Удаляем только реально пустые строки, если они затесались в конце файла
train_df = train_df.dropna()
test_df = test_df.dropna()

# 3. Векторизация (Превращаем слова в числа)
# TF-IDF выделит важные слова-маркеры эмоций
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

# Обучаем векторизатор на тренировочных текстах
X_train = tfidf.fit_transform(train_df['text'])
# Применяем его к тестовым текстам
X_test = tfidf.transform(test_df['text'])

y_train = train_df['emotion']
y_test = test_df['emotion']

# 4. Обучение модели Multinomial Naive Bayes
# Это классика для классификации текстов
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# 5. Оценка результатов
y_pred = nb_model.predict(X_test)

print("\n" + "="*30)
print("РЕЗУЛЬТАТЫ NAIVE BAYES")
print("="*30)
print(f"Общая точность (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print("\nДетальный отчет по метрикам (вставь это в курсовую):")
print(classification_report(y_test, y_pred))

# 6. Проверка на твоем примере
sample = ["I have a beautiful girlfriend"]
sample_vec = tfidf.transform(sample)
pred = nb_model.predict(sample_vec)
print("-" * 30)
print(f"Тест модели: '{sample[0]}'")
print(f"Результат: {pred[0]}")


# 7. Создаем таблицу для сравнения
results_df = pd.DataFrame({
    'Text': test_df['text'],
    'Real_Emotion': y_test,
    'Predicted_Emotion': y_pred
})

# Сохраняем в CSV файл
results_df.to_csv('nb_test_results.csv', index=False, sep=';', encoding='utf-8')
print("\n[ИНФО] Результаты теста сохранены в файл 'nb_test_results.csv'")


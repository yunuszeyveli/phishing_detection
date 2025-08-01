{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNaOazWF6BEUsowF5qmSUua"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "txY7FNPSWhCk",
        "outputId": "2a2d1177-4b46-4e68-970b-181a9a753327"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            " Classification Report:\n",
            "\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       1.00      1.00      1.00         2\n",
            "           1       1.00      1.00      1.00         1\n",
            "\n",
            "    accuracy                           1.00         3\n",
            "   macro avg       1.00      1.00      1.00         3\n",
            "weighted avg       1.00      1.00      1.00         3\n",
            "\n",
            ">>Accuracy<<: 1.0\n",
            ">>Precision<<: 1.0\n",
            ">>Recall<<: 1.0\n",
            ">>F1 Score<<: 1.0\n"
          ]
        }
      ],
      "source": [
        "import re\n",
        "import pandas as pd\n",
        "from urllib.parse import urlparse\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score\n",
        "\n",
        "\n",
        "def extract_features(url):\n",
        "    features = {}\n",
        "    features['url_length'] = len(url)\n",
        "    features['has_ip'] = 1 if re.match(r'\\d+\\.\\d+\\.\\d+\\.\\d+', url) else 0\n",
        "    features['has_at_symbol'] = 1 if '@' in url else 0\n",
        "    features['num_digits'] = sum(char.isdigit() for char in url)\n",
        "    features['has_https'] = 1 if url.startswith('https') else 0\n",
        "    features['count_dots'] = url.count('.')\n",
        "    features['count_hyphens'] = url.count('-')\n",
        "    features['count_slashes'] = url.count('/')\n",
        "    features['has_subdomain'] = 1 if len(urlparse(url).netloc.split('.')) > 2 else 0\n",
        "    return features\n",
        "\n",
        "data = [\n",
        "    {\"url\": \"http://192.168.0.1/phishing\", \"label\": 1},\n",
        "    {\"url\": \"https://www.google.com\", \"label\": 0},\n",
        "    {\"url\": \"http://example.com@evil.com\", \"label\": 1},\n",
        "    {\"url\": \"https://secure-paypal.com.login.account.info\", \"label\": 1},\n",
        "    {\"url\": \"https://bankofamerica.com\", \"label\": 0},\n",
        "    {\"url\": \"http://login.verify-update.com\", \"label\": 1},\n",
        "    {\"url\": \"https://netflix.com\", \"label\": 0},\n",
        "    {\"url\": \"http://malicious.site/secure-banking\", \"label\": 1},\n",
        "    {\"url\": \"https://facebook.com\", \"label\": 0},\n",
        "    {\"url\": \"http://abc.xyz/login?user=test\", \"label\": 1}\n",
        "]\n",
        "\n",
        "df = pd.DataFrame(data)\n",
        "feature_list = [extract_features(url) for url in df['url']]\n",
        "features_df = pd.DataFrame(feature_list)\n",
        "df_final = pd.concat([features_df, df['label']], axis=1)\n",
        "\n",
        "X = df_final.drop('label', axis=1)\n",
        "y = df_final['label']\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n",
        "\n",
        "model = RandomForestClassifier(random_state=42)\n",
        "model.fit(X_train, y_train)\n",
        "y_pred = model.predict(X_test)\n",
        "\n",
        "\n",
        "print(\"\\n Classification Report:\\n\")\n",
        "print(classification_report(y_test, y_pred))\n",
        "\n",
        "print(\">>Accuracy<<:\", accuracy_score(y_test, y_pred))\n",
        "print(\">>Precision<<:\", precision_score(y_test, y_pred))\n",
        "print(\">>Recall<<:\", recall_score(y_test, y_pred))\n",
        "print(\">>F1 Score<<:\", f1_score(y_test, y_pred))"
      ]
    }
  ]
}
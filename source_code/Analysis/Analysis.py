import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")

# Load Data from test resutls
ROBERTA_PATH = './Fact_Checking_model/data/test/Roberta.csv'
BERT_PATH = './Fact_Checking_model/data/test/BERT.csv'
ERRORS_PATH = './Fact_Checking_model/data/analysis/BERT_analysis.xlsx'

df_bert = pd.read_csv(BERT_PATH)
df_roberta = pd.read_csv(ROBERTA_PATH)

df_bert = df_bert.drop(["Unnamed: 0", "Column1"], axis=1)
df_roberta = df_roberta.drop(["Unnamed: 0", "Column1"], axis=1)

#metrics BERT
predictions = df_bert['model_label'].to_numpy()
labels = df_bert['TrueLabel'].to_numpy()

balanced_accuracy_bert = balanced_accuracy_score(labels, predictions)
accuracy_bert = accuracy_score(labels, predictions)

cm_BERT =  confusion_matrix(labels, predictions)

# metrics Roberta
predictions = df_roberta['model_label'].to_numpy()
labels = df_roberta['TrueLabel'].to_numpy()

balanced_accuracy_roberta = balanced_accuracy_score(labels, predictions)
accuracy_roberta = accuracy_score(labels, predictions)

cm_ROBERTA = confusion_matrix(labels, predictions)

# graphs - BERT
class_names = ['Consistent', 'Unrelated', 'Inconsistent']  # Replace with your actual class names

heatmap = sns.heatmap(cm_BERT, annot=True, cmap='Blues', fmt='d', cbar=False, xticklabels=class_names, yticklabels=class_names, annot_kws={'fontsize': 16})

# Iterate over the diagonal elements and customize their colors
plt.xlabel('Predicted Labels', fontsize=16)
plt.ylabel('True Labels', fontsize=16)
plt.title('Confusion Matrix BERT', fontsize=16)
plt.xticks(fontsize=16)  # Set the x-axis font size
plt.yticks(fontsize=16)  # Set the y-axis font size
plt.tight_layout()
plt.show()


# graphs - RoBERTa
#class_names = ['Consistent', 'Unrelated', 'Inconsistent']  # Replace with your actual class names
#sns.heatmap(cm_ROBERTA, annot=True, cmap='Pastel1', fmt='d', cbar=False, xticklabels=class_names, yticklabels=class_names, annot_kws={'fontsize': 16})
#plt.xlabel('Predicted Labels', fontsize=16)
#plt.ylabel('True Labels', fontsize=16)
#plt.title('Confusion Matrix RoBERTa', fontsize=16)
#plt.xticks(fontsize=16)  # Set the x-axis font size
#plt.yticks(fontsize=16)  # Set the y-axis font size
#plt.show()

# Error Analysis - BERT - Inconistencies
counts = [0, 11, 8, 0, 7]
types = ["pronoun swap", "entity swap", "negation", "noise", "doesn't fit any training category"]
data = pd.DataFrame({"Types": types, "Counts": counts})
sns.barplot(x="Types", y="Counts", data=data, palette="viridis")
plt.title("Distribution of Inconsistency Types in Test Data", fontsize=18)
plt.xlabel("Types of inconsistencies", fontsize=18)
plt.ylabel("Counts", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()



#df_errors = df_bert[df_bert['TrueLabel'] != df_bert['model_label']]
#df_errors.to_excel(ERRORS_PATH)
#print(df_errors)

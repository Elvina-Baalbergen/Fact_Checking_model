import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")

# Load Data from test resutls
ROBERTA_PATH = './Fact_Checking_model/data/test/Roberta.csv'
BERT_PATH = './Fact_Checking_model/data/test/BERT.csv'
ERRORS_PATH = './Fact_Checking_model/data/analysis/BERT_analysis.xlsx'
ALL_ICONSISTENT_BERT_PATH = './Fact_Checking_model/data/analysis/BERT_analysis_ALL.xlsx'
ALL_ICONSISTENT_BERT_PATH_ANNOTATED = './Fact_Checking_model/data/analysis/BERT_analysis_ALL_annotated.xlsx'
ALL_UNRELATED_BUT_CONSISTEND = './Fact_Checking_model/data/analysis/BERT_analysis_ALL_CbU_annotated.xlsx'

df_bert = pd.read_csv(BERT_PATH)
df_roberta = pd.read_csv(ROBERTA_PATH)

df_bert = df_bert.drop(["Unnamed: 0"], axis=1)
df_roberta = df_roberta.drop(["Unnamed: 0"], axis=1)

# export for analysis
df_errors = df_bert[df_bert['TrueLabel'] != df_bert['model_label']]
df_errors.to_excel(ERRORS_PATH)
df_inconsistent = df_bert[df_bert['TrueLabel'] == 2]
df_inconsistent.to_excel(ALL_ICONSISTENT_BERT_PATH)

df_unrelated = df_bert[df_bert['TrueLabel'] == 0]
df_unrelated_but_consistend = df_unrelated[df_unrelated['model_label'] == 1]
df_unrelated_but_consistend.to_excel(ALL_UNRELATED_BUT_CONSISTEND)

# Import annotated for analysis
df_inconsistent_bert = pd.read_excel(ALL_ICONSISTENT_BERT_PATH_ANNOTATED)

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
# inconsistices performence per class
classes = ["Pronoun swap", "Entity swap", "Negation", "Noise", "Other"]

sum_pronouns_swaps =  df_inconsistent_bert["Pronoun swap"].sum()
sum_entity_swaps =  df_inconsistent_bert["Entity swap"].sum()
sum_negation =  df_inconsistent_bert["Negation"].sum()
sum_Noise =  df_inconsistent_bert["Noise"].sum()
sum_other =  df_inconsistent_bert["Other"].sum()
total_nr = [sum_pronouns_swaps, sum_entity_swaps, sum_negation, sum_Noise, sum_other]

df_inconsistent_bert_correct = df_inconsistent_bert[df_inconsistent_bert['TrueLabel'] == df_inconsistent_bert['model_label']]
sum_pronouns_swaps_correct =  df_inconsistent_bert_correct["Pronoun swap"].sum()
sum_entity_swaps_correct =  df_inconsistent_bert_correct["Entity swap"].sum()
sum_negation_correct =  df_inconsistent_bert_correct["Negation"].sum()
sum_Noise_correct =  df_inconsistent_bert_correct["Noise"].sum()
sum_other_correct =  df_inconsistent_bert_correct["Other"].sum()
total_nr_correct = [sum_pronouns_swaps_correct, sum_entity_swaps_correct, sum_negation_correct, sum_Noise_correct, sum_other_correct]

data = pd.DataFrame({'Classes': classes, 'Total': total_nr, "Correct":total_nr_correct})

sns.barplot(x='Classes', y='Total', data=data, color='grey', alpha=0.5, label='Total')
sns.barplot(x='Classes', y='Correct', data=data, color='green', alpha=0.7, label='Correct')

plt.xlabel('Types', fontsize= 20)
plt.ylabel('Count', fontsize= 20)
plt.xticks(fontsize=20)  
plt.yticks(fontsize=20) 
plt.title('Total number of inconsistencies by type', fontsize=26)
plt.legend(fontsize=20)
plt.show()

# Confusion matrix
class_names = ['Consistent', 'Unrelated', 'Inconsistent']  # Replace with your actual class names
sns.heatmap(cm_BERT, annot=True, cmap='Blues', fmt='d', cbar=False, xticklabels=class_names, yticklabels=class_names, annot_kws={'fontsize': 16})
plt.xlabel('Predicted Labels', fontsize=20)
plt.ylabel('True Labels', fontsize=20)
plt.title('Confusion Matrix BERT', fontsize=16)
plt.xticks(fontsize=20)  
plt.yticks(fontsize=20)  
plt.tight_layout()
plt.show()

'''
# graphs - RoBERTa
class_names = ['Consistent', 'Unrelated', 'Inconsistent']  # Replace with your actual class names
sns.heatmap(cm_ROBERTA, annot=True, cmap='Blues', fmt='d', cbar=False, xticklabels=class_names, yticklabels=class_names, annot_kws={'fontsize': 16})
plt.xlabel('Predicted Labels', fontsize=16)
plt.ylabel('True Labels', fontsize=16)
plt.title('Confusion Matrix RoBERTa', fontsize=16)
plt.xticks(fontsize=16)  
plt.yticks(fontsize=16)  
plt.show()
'''
# Plot vs Reviews
print(df_bert)
df_bert_plot = df_bert[df_bert["Type"] == "plot"]
df_bert_review = df_bert[df_bert["Type"] == "review"]

predictions_plot = df_bert_plot['model_label'].to_numpy()
labels_plot = df_bert_plot['TrueLabel'].to_numpy()
predictions_review = df_bert_review['model_label'].to_numpy()
labels_review  = df_bert_review['TrueLabel'].to_numpy()

balanced_accuracy_plot= balanced_accuracy_score(labels_plot, predictions_plot)
accuracy_plot = accuracy_score(labels_plot, predictions_plot)
f1_plot = f1_score(labels_plot, predictions_plot, average='macro')

balanced_accuracy_review = balanced_accuracy_score(labels_review, predictions_review)
accuracy_reviews = accuracy_score(labels_review, predictions_review)
f1_reviews = f1_score(labels_review, predictions_review, average='macro')

print([f1_plot, accuracy_plot, balanced_accuracy_plot, f1_reviews, accuracy_reviews, balanced_accuracy_review])
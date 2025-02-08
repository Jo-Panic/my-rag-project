# prompts.py

"""
Ce fichier contient tous les prompts utilisés dans le système RAG.
Les prompts sont organisés par fonction et incluent des commentaires explicatifs.

This file contains all the prompts used in the RAG system.
Prompts are organized by function and include explanatory comments.
"""


def get_validation_prompt(question: str, context: str) -> str:
    """
    Generates the prompt for validating passage relevance.

    Args:
        question: The user's question
        context: The retrieved text passages

    Returns:
        str: The formatted validation prompt
    """
    return f"""INSTRUCTION : Vous êtes un vérificateur qui valide si les documents fournis contiennent les informations nécessaires pour répondre à la question.

Question : {question}

Documents à vérifier :
{context}

RÈGLES DE VALIDATION :
1. Pour les questions sur "comment faire" ou les tutoriels :
   - Les documents doivent contenir des instructions détaillées
   - Un simple exemple n'est pas suffisant

2. Pour les questions sur des tâches ou procédures :
   - Les documents peuvent contenir les informations sous différentes sections
   - Les horaires spécifiques et "selon disponibilités" sont considérés comme pertinents
   - Une liste de tâches reliées à la période demandée est suffisante

3. Pour les questions sur des définitions ou concepts :
   - Les documents doivent contenir explicitement l'information
   - Les mentions indirectes ne sont pas suffisantes

4. Pour les questions générales :
   - Si les documents contiennent des informations partielles mais utiles : répondez OUI
   - Si les documents ne contiennent aucune information pertinente : répondez NON

La question peut-elle être répondue avec les informations des documents (OUI ou NON) ?"""


def get_response_prompt() -> str:
    """
    Returns the system prompt for response generation.

    Returns:
        str: The system prompt for response generation
    """
    return """INSTRUCTION : Vous êtes un assistant qui répond aux questions en utilisant les informations des documents fournis.

RÈGLES IMPORTANTES :
1. Pour les questions techniques ou conceptuelles :
   - Ne fournissez QUE les informations explicitement dans les documents
   - Ne complétez PAS avec vos connaissances

2. Pour les questions sur des tâches ou procédures :
   - Rassemblez les informations pertinentes de différentes sections
   - Organisez la réponse de manière chronologique si possible
   - Mentionnez les conditions (horaires spécifiques, selon disponibilités)

3. Pour toute réponse :
   - Restez fidèle au niveau de détail des documents
   - N'ajoutez pas d'interprétations personnelles
   - Si l'information est incomplète, précisez-le

4. Si vous ne pouvez pas répondre : dites-le clairement

Répondez maintenant à la question en suivant ces règles :"""

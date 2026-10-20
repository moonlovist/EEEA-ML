"""Gentle feedback for small independent exercises."""
import math

def check_answer(value, expected, lang='en', tolerance=1e-7):
    if value is None:
        print('Replace None with your answer, then run this cell again.' if lang=='en' else 'Remplacer None par votre réponse, puis relancer cette cellule.')
        return 'pending'
    if isinstance(expected,(int,float)):
        good=type(value) in (int,float) and math.isclose(value,expected,abs_tol=tolerance,rel_tol=tolerance)
    else:good=value==expected
    if good:
        print('Correct. Explain the calculation in one sentence.' if lang=='en' else 'Correct. Expliquer le calcul en une phrase.')
    else:
        print('Compare each step with Hint 1, then try again.' if lang=='en' else 'Comparer chaque étape à l’indice 1, puis réessayer.')
    return 'correct' if good else 'retry'

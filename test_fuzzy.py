from fuzzy import calculate_reliability

confidence = 90

score = calculate_reliability(confidence)

print("AI Confidence:", confidence, "%")
print("Fuzzy Reliability:", round(score, 2), "%")
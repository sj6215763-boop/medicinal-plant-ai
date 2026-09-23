import sqlite3

conn = sqlite3.connect("plants.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS plants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    common_name TEXT,
    scientific_name TEXT,
    description TEXT,
    traditional_uses TEXT,
    precautions TEXT
)
""")

plants = [
    (
        "Tulsi",
        "Ocimum tenuiflorum",
        "An aromatic herb commonly grown in India.",
        "Traditionally used in various Indian systems of traditional medicine.",
        "Traditional use does not establish safety or effectiveness for treating disease."
    ),

    (
        "Neem",
        "Azadirachta indica",
        "A tree widely found in India.",
        "Neem has a long history of traditional use.",
        "Use should be based on appropriate safety guidance."
    ),

    (
        "Aloe Vera",
        "Aloe vera",
        "A succulent plant with fleshy leaves.",
        "Traditionally used for various purposes including skin-related applications.",
        "Different aloe preparations can have different safety considerations."
    ),

    (
        "Mint",
        "Mentha",
        "An aromatic herb.",
        "Commonly used as a culinary and traditional herb.",
        "Concentrated preparations may have different safety considerations."
    ),

    (
        "Hibiscus",
        "Hibiscus rosa-sinensis",
        "A flowering plant commonly grown in India.",
        "Used traditionally in several cultural and herbal practices.",
        "Traditional use should not be considered a substitute for medical treatment."
    )
]

cursor.executemany("""
INSERT INTO plants
(common_name, scientific_name, description, traditional_uses, precautions)
VALUES (?, ?, ?, ?, ?)
""", plants)

conn.commit()
conn.close()

print("Database created successfully!")
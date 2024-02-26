import sqlite3

class DatabaseManager:

    def sets_up_table_Board(self):
        connection = sqlite3.connect('Settlers_Of_Catan.db') # Connect to the SQLite database (or create it if it doesn't exist)
        cursor = connection.cursor()
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS Board (
                ID TEXT PRIMARY KEY,
                CN10 TEXT, CN9 TEXT, CN8 TEXT, CN7 TEXT, CN6 TEXT, 
                CN5 TEXT, CN4 TEXT, CN3 TEXT, CN2 TEXT, CN1 TEXT, 
                C0 TEXT, C1 TEXT, C2 TEXT, C3 TEXT, C4 TEXT, 
                C5 TEXT, C6 TEXT, C7 TEXT, C8 TEXT, C9 TEXT, C10 TEXT
            )
            '''
        )
        for ID in ['RN10', 'RN9', 'RN8', 'RN7', 'RN6', 'RN5', 'RN4', 'RN3', 'RN2', 'RN1', 'R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10']:
            cursor.execute(f"SELECT EXISTS(SELECT 1 FROM Board WHERE ID='{ID}' LIMIT 1)")
            row_with_ID_exists = cursor.fetchone()[0]
            if not row_with_ID_exists:
                cursor.execute(
                    f'''
                    INSERT INTO
                    Board (ID, CN10, CN9, CN8, CN7, CN6, CN5, CN4, CN3, CN2, CN1, C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10)
                    VALUES ('{ID}', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N')
                    '''
                )
        connection.commit()
        connection.close()
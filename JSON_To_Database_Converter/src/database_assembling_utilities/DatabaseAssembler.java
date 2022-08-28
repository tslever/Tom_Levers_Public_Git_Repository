package database_assembling_utilities;


import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import org.json.JSONArray;
import org.json.JSONObject;
import org.sqlite.SQLiteConfig;


public class DatabaseAssembler {

	
	public static void createDatabase(String pathToDatabase) throws SQLException {
		Connection connection = DriverManager.getConnection("jdbc:sqlite:" + pathToDatabase);
		connection.close();
	}
	
	
	public static void createTables(String pathToDatabase) throws SQLException {
		
		Connection connection = DriverManager.getConnection("jdbc:sqlite:" + pathToDatabase);		
		Statement statement = connection.createStatement();
		
		String query =
			"CREATE TABLE IF NOT EXISTS Products (\n" +
			"id_of_product INTEGER NOT NULL PRIMARY KEY UNIQUE,\n" +
			"name TEXT NOT NULL UNIQUE,\n" +
			"image_closed TEXT,\n" +
			"image_open TEXT,\n" +
			"description TEXT,\n" +
			"story TEXT,\n" +
			"productId INTEGER NOT NULL UNIQUE\n" +
			");";
		statement.execute(query);
		
		query =
			"CREATE TABLE IF NOT EXISTS Ingredients (\n" +
			"id_of_ingredient INTEGER NOT NULL PRIMARY KEY UNIQUE,\n" +
			"value_of_ingredient TEXT NOT NULL UNIQUE\n" +
			");";
		statement.execute(query);
		
		query =
			"CREATE TABLE IF NOT EXISTS SourcingValues (\n" +
			"id_of_sourcing_value INTEGER NOT NULL PRIMARY KEY UNIQUE,\n" +
			"value_of_sourcing_value TEXT NOT NULL UNIQUE\n" +
			");";
		statement.execute(query);
		
		query =
			"CREATE TABLE IF NOT EXISTS AllergyInfos (\n" +
			"id_of_allergy_info INTEGER NOT NULL PRIMARY KEY UNIQUE,\n" +
			"value_of_allergy_info TEXT NOT NULL UNIQUE" +
			");";
		statement.execute(query);
		
		query =
			"CREATE TABLE IF NOT EXISTS DietaryCertifications (\n" +
			"id_of_dietary_certification NOT NULL PRIMARY KEY UNIQUE,\n" +
			"value_of_dietary_certification TEXT NOT NULL UNIQUE\n" +
			");";
		statement.execute(query);
		
		query =
			"CREATE TABLE IF NOT EXISTS AssociationsBetweenProductsAndIngredients (\n" +
			"id_of_association_between_product_and_ingredient INTEGER NOT NULL PRIMARY KEY UNIQUE,\n" +
			"index_of_product INTEGER NOT NULL,\n" +
			"index_of_ingredient INTEGER NOT NULL,\n" +
			"FOREIGN KEY(index_of_product) REFERENCES Products(id_of_product),\n" +
			"FOREIGN KEY(index_of_ingredient) REFERENCES Ingredients(id_of_ingredient)\n" +
			");";
		statement.execute(query);
		
		query =
			"CREATE TABLE IF NOT EXISTS AssociationsBetweenProductsAndSourcingValues (\n" +
			"id_of_association_between_product_and_sourcing_value INTEGER NOT NULL PRIMARY KEY UNIQUE,\n" +
			"index_of_product INTEGER NOT NULL,\n" +
			"index_of_sourcing_value INTEGER NOT NULL,\n" +
			"FOREIGN KEY(index_of_product) REFERENCES Products(id_of_product),\n" +
			"FOREIGN KEY(index_of_sourcing_value) REFERENCES SourcingValues(id_of_sourcing_value)\n" +
			");";
		statement.execute(query);
		
		query =
			"CREATE TABLE IF NOT EXISTS AssociationsBetweenProductsAndAllergyInfos (\n" +
			"id_of_association_between_product_and_allergy_info INTEGER NOT NULL PRIMARY KEY UNIQUE,\n" +
			"index_of_product INTEGER NOT NULL,\n" +
			"index_of_allergy_info INTEGER NOT NULL,\n" +
			"FOREIGN KEY(index_of_product) REFERENCES Products(id_of_product),\n" +
			"FOREIGN KEY(index_of_allergy_info) REFERENCES AllergyInfos(id_of_allergy_info)\n" +
			");";
		statement.execute(query);
		
		query =
			"CREATE TABLE IF NOT EXISTS AssociationsBetweenProductsAndDietaryCertifications (\n" +
			"id_of_association_between_product_and_dietary_certification INTEGER NOT NULL PRIMARY KEY UNIQUE,\n" +
			"index_of_product INTEGER NOT NULL,\n" +
			"index_of_dietary_certification INTEGER NOT NULL,\n" +
			"FOREIGN KEY(index_of_product) REFERENCES Products(id_of_product),\n" +
			"FOREIGN KEY(index_of_dietary_certification) " +
			"REFERENCES DietaryCertifications(id_of_dietary_certification)\n" +
			");";
		
		statement.close();
		connection.close();
		
	}
	
	
	public static void insert(String contentsOfFile, String pathToDatabase) throws SQLException {
		
		contentsOfFile = contentsOfFile.replaceAll("'", "''");
		
		SQLiteConfig sqliteConfig = new SQLiteConfig();
		sqliteConfig.enforceForeignKeys(true);
		Connection connection = DriverManager.getConnection(
			"jdbc:sqlite:" + pathToDatabase, sqliteConfig.toProperties());
		Statement statement = connection.createStatement();
		
		JSONObject contentsOfFileAsJSONObject = new JSONObject(contentsOfFile);
		JSONArray jsonArray = contentsOfFileAsJSONObject.getJSONArray("icecreamproducts");
		
		JSONObject product;
		String query;
		for (int i = 0; i < jsonArray.length(); i++) {
			
			product = jsonArray.getJSONObject(i);
			
			query =
				"INSERT INTO Products " +
				"(id_of_product, name, image_closed, image_open, description, story, productId)\n" +
				"VALUES (" +
				i + ", " +
				"'" + product.getString("name") + "', " +
				"'" + product.getString("image_closed") + "', " +
				"'" + product.getString("image_open") + "', " +
				"'" + product.getString("description") + "', " +
				"'" + product.getString("story") + "', " +
				product.getInt("productId") +
				")\n" +
				"ON CONFLICT(id_of_product) DO UPDATE SET id_of_product = id_of_product;";
			
			statement.execute(query);
		}
		
		query = "SELECT * FROM PRODUCTS;";
		ResultSet resultSet = statement.executeQuery(query);
		while (resultSet.next()) {
			System.out.println(
				resultSet.getInt("id_of_product") + " | " +
				resultSet.getString("name") + " | " +
				resultSet.getString("image_closed") + " | " +
				resultSet.getString("image_open") + " | " +
				resultSet.getString("description") + " | " +
				resultSet.getString("story") + " | " +
				resultSet.getString("productId")
			);
		}
		
		statement.close();
		connection.close();
		
	}
	
	
}

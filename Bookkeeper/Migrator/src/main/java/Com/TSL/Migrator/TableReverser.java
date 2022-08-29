package Com.TSL.Migrator;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.Stack;
import java.util.regex.Pattern;

public class TableReverser {

	public static void main(String[] args) throws SQLException {
		try (
			Connection connection = DriverManager.getConnection("jdbc:postgresql://localhost:5432/accounts_database", "postgres", "password");
			Statement statement = connection.createStatement();
		) {
	        Stack<LineItem> stack = new Stack<>();
			String query = "SELECT * FROM direct_deposit_and_chk";
			ResultSet resultSet = statement.executeQuery(query);			
			while (resultSet.next()) {
	            LineItem lineItem = new LineItem(
	                resultSet.getInt("id"),
	                resultSet.getString("date"),
	                resultSet.getString("name"),
	                resultSet.getString("account_associated_with_value"),
	                resultSet.getString("complementary_account"),
	                resultSet.getFloat("value")
	            );
	        	System.out.println(lineItem);
	            stack.push(lineItem);
	        }
			
			query = "DELETE FROM direct_deposit_and_chk";
			statement.execute(query);
	        
			System.out.println(stack);
			
			int id = 0;
			while (!stack.isEmpty()) {
				LineItem lineItem = stack.pop();
		        query =
	                "INSERT INTO direct_deposit_and_chk\n" +
	                "(id, date, name, account_associated_with_value, complementary_account, value)\n" +
	                "VALUES ('" +
	                	String.valueOf(id).replaceAll(Pattern.quote("|"), ",") + "', '" +
	                    lineItem.date().replaceAll(Pattern.quote("|"), ",") + "', '" +
	                    lineItem.name().replaceAll(Pattern.quote("|"), ",") + "', '" +
	                    lineItem.accountAssociatedWithValue().replaceAll(Pattern.quote("|"), ",") + "', '" +
	                    lineItem.complementaryAccount().replaceAll(Pattern.quote("|"), ",") + "', '" +
	                    String.format("%.2f", lineItem.value()).replaceAll(Pattern.quote("|"), ",") +
	                "');";
		        statement.execute(query);
		        System.out.println(lineItem);
		        id++;
			}
		}
	}

}

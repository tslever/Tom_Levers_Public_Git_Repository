package Com.TSL.Bookkeeper;

import java.util.Date;

import javafx.beans.property.SimpleFloatProperty;
import javafx.beans.property.SimpleIntegerProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.beans.property.SimpleStringProperty;

public class LineItem {
	
	private SimpleIntegerProperty id;
	private SimpleObjectProperty<Date> date;
	private SimpleStringProperty name;
	private SimpleObjectProperty<Account> accountAssociatedWithValue;
	private SimpleObjectProperty<Account> complementaryAccount;
	private SimpleFloatProperty value;
	
	public LineItem(int id, Date date, String name, Account accountAssociatedWithValue, Account complementaryAccount, float value) {
		this.id = new SimpleIntegerProperty(id);
		this.date = new SimpleObjectProperty<Date>(date);
		this.name = new SimpleStringProperty(name);
		this.accountAssociatedWithValue = new SimpleObjectProperty<Account>(accountAssociatedWithValue);
		this.complementaryAccount = new SimpleObjectProperty<Account>(complementaryAccount);
		this.value = new SimpleFloatProperty(value);
	}
	
	public SimpleIntegerProperty getId() {
		return this.id;
	}
	
	public SimpleObjectProperty<Date> getDate() {
		return this.date;
	}
	
	public void setDate(Date date) {
		this.date = new SimpleObjectProperty<Date>(date);
	}
	
	public SimpleStringProperty getName() {
		return this.name;
	}
	
	public void setName(String name) {
		this.name = new SimpleStringProperty(name);
	}
	
	public SimpleObjectProperty<Account> getAccountAssociatedWithValue() {
		return this.accountAssociatedWithValue;
	}
	
	public void setAccountAssociatedWithValue(Account accountAssociatedWithValue) {
		this.accountAssociatedWithValue = new SimpleObjectProperty<Account>(accountAssociatedWithValue);
	}

	public SimpleObjectProperty<Account> getComplementaryAccount() {
		return this.complementaryAccount;
	}
	
	public void setComplementaryAccount(Account complementaryAccount) {
		this.complementaryAccount = new SimpleObjectProperty<Account>(complementaryAccount);
	}
	
	public SimpleFloatProperty getValue() {
		return this.value;
	}
	
	public void setValue(float value) {
		this.value = new SimpleFloatProperty(value);
	}
	
	@Override
	public String toString() {
		return "{" +
			this.id.get() + ", " +
			this.date.getValue() + ", " +
			this.name.getValue() + ", " +
			this.accountAssociatedWithValue.getValue() + ", " +
			this.complementaryAccount.getValue() + ", " +
			this.value.get() +
		"}";
	}
}
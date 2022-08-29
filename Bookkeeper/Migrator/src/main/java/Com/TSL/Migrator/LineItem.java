package Com.TSL.Migrator;

public class LineItem {

	private int id;
	private String date;
	private String name;
	private String accountAssociatedWithValue;
	private String complementaryAccount;
	private float value;
	
	public LineItem(int id, String date, String name, String accountAssociatedWithValue, String complementaryAccount, float value) {
		this.id = id;
		this.date = date;
		this.name = name;
		this.accountAssociatedWithValue = accountAssociatedWithValue;
		this.complementaryAccount = complementaryAccount;
		this.value = value;
	}
	
	public int id() {
		return id;
	}
	
	public String date() {
		return date;
	}
	
	public String name() {
		return name;
	}
	
	public String accountAssociatedWithValue() {
		return accountAssociatedWithValue;
	}
	
	public String complementaryAccount() {
		return complementaryAccount;
	}
	
	public float value() {
		return value;
	}
	
	@Override
	public String toString() {
		return
			"[" +
			this.id + "," +
			this.date + "," +
			this.name + "," +
			this.accountAssociatedWithValue + "," +
			this.complementaryAccount + "," +
			this.value +
			"]";
	}
	
}

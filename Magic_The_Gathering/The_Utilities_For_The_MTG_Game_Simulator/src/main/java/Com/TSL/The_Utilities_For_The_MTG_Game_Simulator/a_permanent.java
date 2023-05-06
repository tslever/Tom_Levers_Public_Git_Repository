package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;


public abstract class a_permanent {

	private String Name;
	private boolean Tapped_Status;
	
	public a_permanent(String The_Name_To_Use, boolean The_Tapped_Status_To_Use) {
		this.Name = The_Name_To_Use;
		this.Tapped_Status = The_Tapped_Status_To_Use;
	}
	
	public boolean indicates_whether_it_is_tapped() {
		return this.Tapped_Status;
	}
	
	public String provides_its_name() {
		return this.Name;
	}
	
	public void untaps() {
		this.Tapped_Status = false;
	}
	
	@Override
	public String toString() {
		return this.Name;
	}
}
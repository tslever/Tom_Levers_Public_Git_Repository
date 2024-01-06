package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

public class an_activated_ability extends an_ability {

	private String Cost;
	private boolean Indicator_Of_Whether_It_Is_Activatable;
	
	public an_activated_ability(String The_Cost_To_Use, String The_Effect_To_Use, a_permanent The_Permanent_To_Use) {
		super(The_Effect_To_Use, The_Permanent_To_Use);
		this.Cost = The_Cost_To_Use;
		if (this.Cost.equals("T")) {
			if (this.permanent().is_tapped()) {
				this.Indicator_Of_Whether_It_Is_Activatable = false;
			} else {
				this.Indicator_Of_Whether_It_Is_Activatable = true;
			}
		}
	}
	
	public void becomes_activatable() {
		this.Indicator_Of_Whether_It_Is_Activatable = true;
	}
	
	public void becomes_nonactivatable() {
		this.Indicator_Of_Whether_It_Is_Activatable = false;
	}
	
	public boolean is_activatable() {
		return this.Indicator_Of_Whether_It_Is_Activatable;
	}
	
	public String cost() {
		return this.Cost;
	}
	
	public boolean requires_tapping() {
		return true;
	}
	
	@Override
	public String toString() {
		return this.permanent().name() + ": " + this.Cost + ": " + this.effect();
	}
}

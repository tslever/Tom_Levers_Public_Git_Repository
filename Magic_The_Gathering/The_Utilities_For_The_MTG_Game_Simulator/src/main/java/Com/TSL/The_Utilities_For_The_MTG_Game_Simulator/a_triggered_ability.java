package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

public class a_triggered_ability extends an_ability {

	private String Event;
	
	public a_triggered_ability(String The_Event_To_Use, String The_Effect_To_Use, a_permanent The_Permanent_To_Use) {
		super(The_Effect_To_Use, The_Permanent_To_Use);
		this.Event = The_Event_To_Use;
	}
	
	public String event() {
		return this.Event;
	}
	
	@Override
	public String toString() {
		return this.permanent().player() + ": " + this.permanent().provides_its_name() + ": When " + this.Event + ", " + this.effect();
	}
}

package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

public class a_Plains extends a_land {

	public a_Plains() {
		super("Plains");
		a_mana_ability The_Mana_Ability = new a_mana_ability("T", "Add [W].", this);
		this.receives(The_Mana_Ability);
	}
	
}

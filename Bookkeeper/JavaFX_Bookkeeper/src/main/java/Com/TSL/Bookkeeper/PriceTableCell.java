package Com.TSL.Bookkeeper;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.text.NumberFormat;

import javafx.scene.control.ContentDisplay;
import javafx.scene.control.Label;
import javafx.scene.control.TableCell;
import javafx.scene.control.cell.TextFieldTableCell;
import javafx.scene.layout.AnchorPane;
import javafx.util.converter.FloatStringConverter;


/**
 * PriceTableCell
 * @author James_D, https://stackoverflow.com/questions/48552499/accounting-style-table-cell-in-javafx
 *
 * @param <T>
 */

public class PriceTableCell<T> extends TextFieldTableCell<T, Float> {

    private final AnchorPane pane ;
    private final Label valueLabel ;
    private DecimalFormat format;

    public PriceTableCell() {
    	this.setConverter(new FloatStringConverter());
        format = (DecimalFormat) NumberFormat.getCurrencyInstance();
        format.applyPattern("#,##0.00;(#,##0.00)");
        String symbol = format.getCurrency().getSymbol();
        DecimalFormatSymbols symbols = format.getDecimalFormatSymbols();
        symbols.setCurrencySymbol("");
        format.setDecimalFormatSymbols(symbols);
        Label currencySignLabel = new Label(symbol);
        valueLabel = new Label();
        pane = new AnchorPane(currencySignLabel, valueLabel);
        AnchorPane.setLeftAnchor(currencySignLabel, 0.0);
        AnchorPane.setRightAnchor(valueLabel, 0.0);
        setContentDisplay(ContentDisplay.GRAPHIC_ONLY);
    }

    @Override
    public void updateItem(Float price, boolean empty) {
        super.updateItem(price, empty);
        if (empty) {
            setGraphic(null);
        } else {
            valueLabel.setText(format.format(price));
            setGraphic(pane);
        }
    }
}
package Com.TSL.Bookkeeper;

import java.sql.SQLException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.Date;

import javafx.application.Application;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.geometry.Rectangle2D;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart.Data;
import javafx.scene.chart.XYChart.Series;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.DatePicker;
import javafx.scene.control.Label;
import javafx.scene.control.Tab;
import javafx.scene.control.TabPane;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableColumn.CellDataFeatures;
import javafx.scene.control.TableColumn.CellEditEvent;
import javafx.scene.control.TableView;
import javafx.scene.control.TextField;
import javafx.scene.control.cell.ComboBoxTableCell;
import javafx.scene.control.cell.TextFieldTableCell;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.Priority;
import javafx.scene.layout.RowConstraints;
import javafx.scene.layout.VBox;
import javafx.stage.Screen;
import javafx.stage.Stage;
import javafx.util.converter.FloatStringConverter;

public class Bookkeeper extends Application {

    @Override
    public void start(Stage stage) throws ParseException, SQLException {
    	Rectangle2D primaryScreenBounds = Screen.getPrimary().getBounds();
    	
    	stage.setWidth(primaryScreenBounds.getWidth());
    	stage.setHeight(primaryScreenBounds.getHeight());
        stage.setTitle("Accounter");
        
        GridPane gridPaneForSubmitLineItem = new GridPane();
        //gridPaneForSubmitLineItem.add(new Label("Submit LineItem"), 0, 0);
        gridPaneForSubmitLineItem.add(new Label("Date"), 1, 0);
        gridPaneForSubmitLineItem.add(new Label("Name"), 2, 0);
        gridPaneForSubmitLineItem.add(new Label("Account Associated With Value"), 3, 0);
        gridPaneForSubmitLineItem.add(new Label("Complementary Account"), 4, 0);
        gridPaneForSubmitLineItem.add(new Label("Value"), 5, 0);
        
        DatePicker datePicker = new DatePicker();
        gridPaneForSubmitLineItem.add(datePicker, 1, 1);
        
        TextField nameTextField = new TextField();
        gridPaneForSubmitLineItem.add(nameTextField, 2, 1);

        ComboBox<Account> accountAssociatedWithValue = new ComboBox<Account>(FXCollections.observableArrayList(Account.values()));
        gridPaneForSubmitLineItem.add(accountAssociatedWithValue, 3, 1);
        
        ComboBox<Account> complementaryAccount = new ComboBox<Account>(FXCollections.observableArrayList(Account.values()));
        gridPaneForSubmitLineItem.add(complementaryAccount, 4, 1);

        TextField valueTextField = new TextField();
        gridPaneForSubmitLineItem.add(valueTextField, 5, 1);
        
    	ObservableList<LineItem> observableList = (new DatabaseManager()).getLineItems("Direct_Deposit_And_CHK");

        TableColumn<LineItem, Integer> idColumn = new TableColumn<>("ID");
        idColumn.setCellValueFactory((CellDataFeatures<LineItem, Integer> cellDataFeatures) -> { return cellDataFeatures.getValue().getId().asObject(); }); // TODO: Change lambda expressions to object.

        TableColumn<LineItem, Date> dateColumn = new TableColumn<>("Date");
        dateColumn.setCellValueFactory((CellDataFeatures<LineItem, Date> cellDataFeatures) -> { return cellDataFeatures.getValue().getDate(); });
        dateColumn.setCellFactory((TableColumn<LineItem, Date> tableColumn) -> { return new DateEditingCell(); });

        dateColumn.setOnEditCommit(
            new EventHandler<CellEditEvent<LineItem, Date>>() {
    			@Override
    			public void handle(CellEditEvent<LineItem, Date> event) {
    				int rowIndex = event.getTablePosition().getRow();
    				Date oldDate = observableList.get(rowIndex).getDate().getValue();
    				event.getRowValue().setDate(event.getNewValue());
    				event.getTableView().refresh();
    				Date newDate = observableList.get(rowIndex).getDate().getValue();
    				System.out.println("Changed date from " + new SimpleDateFormat("yyyy-MM-dd").format(oldDate) + " to " + new SimpleDateFormat("yyyy-MM-dd").format(newDate) + ".");
    				// TODO: update date in database.
    				System.out.println("TODO: Update date in database.");
    			}
    		}                
        );
        
//        // The following code block acts as the above code block.
//        dateColumn.setOnEditCommit((CellEditEvent<LineItem, Date> event) -> {
//			int rowIndex = event.getTablePosition().getRow();
//			Date oldDate = observableList.get(rowIndex).getDate().getValue();
//			event.getRowValue().setDate(event.getNewValue());
//			event.getTableView().refresh();
//			Date newDate = observableList.get(rowIndex).getDate().getValue();
//			System.out.println("Changed date from " + new SimpleDateFormat("yyyy-MM-dd").format(oldDate) + " to " + new SimpleDateFormat("yyyy-MM-dd").format(newDate) + ".");
//			// TODO: update date in database.
//			System.out.println("TODO: Update date in database.");
//        });

    	TableColumn<LineItem, String> nameColumn = new TableColumn<>("Name");
    	nameColumn.setCellValueFactory((CellDataFeatures<LineItem, String> cellDataFeatures) -> { return cellDataFeatures.getValue().getName(); });
    	nameColumn.setCellFactory(TextFieldTableCell.forTableColumn());     
    	nameColumn.setOnEditCommit(
    		new EventHandler<CellEditEvent<LineItem, String>>() {
    			@Override
    			public void handle(CellEditEvent<LineItem, String> event) {
    				int rowIndex = event.getTablePosition().getRow();
    				String oldName = observableList.get(rowIndex).getName().getValue();
    				event.getRowValue().setName(event.getNewValue());
    				event.getTableView().refresh();
    				String newName = observableList.get(rowIndex).getName().getValue();
    				System.out.println("Changed name from \"" + oldName + "\" to \"" + newName + "\".");
    				// TODO: update name in database.
    				System.out.println("TODO: update name in database.");
    			}
    		}
    	);
    	
    	TableColumn<LineItem, Account> accountAssociatedWithValueColumn = new TableColumn<>("Account Associated With Value");
    	accountAssociatedWithValueColumn.setCellValueFactory((CellDataFeatures<LineItem, Account> cellDataFeatures) -> { return cellDataFeatures.getValue().getAccountAssociatedWithValue(); });
    	ObservableList<Account> accountsAssociatedWithValue = FXCollections.observableArrayList(Account.values());
    	accountAssociatedWithValueColumn.setCellFactory(ComboBoxTableCell.<LineItem, Account>forTableColumn(accountsAssociatedWithValue));
    	accountAssociatedWithValueColumn.setOnEditCommit(
    		new EventHandler<CellEditEvent<LineItem, Account>>() {
    			@Override
    			public void handle(CellEditEvent<LineItem, Account> event) {
    				int rowIndex = event.getTablePosition().getRow();
    				Account oldAccount = observableList.get(rowIndex).getAccountAssociatedWithValue().getValue();
    				event.getRowValue().setAccountAssociatedWithValue(event.getNewValue());
    				event.getTableView().refresh();
    				Account newAccount = observableList.get(rowIndex).getAccountAssociatedWithValue().getValue();
    				System.out.println("Changed account associated with value from " + oldAccount + " to " + newAccount + ".");
    				// TODO: update account associated with value in database.
    				System.out.println("todo: update account associated with value in database.");
    			}
    		}
    	);
    	
    	TableColumn<LineItem, Account> complementaryAccountColumn = new TableColumn<>("Complementary Account");
    	complementaryAccountColumn.setCellValueFactory((CellDataFeatures<LineItem, Account> cellDataFeatures) -> { return cellDataFeatures.getValue().getComplementaryAccount(); });
    	ObservableList<Account> complementaryAccounts = FXCollections.observableArrayList(Account.values());
    	accountAssociatedWithValueColumn.setCellFactory(ComboBoxTableCell.<LineItem, Account>forTableColumn(complementaryAccounts));
    	accountAssociatedWithValueColumn.setOnEditCommit(
    		new EventHandler<CellEditEvent<LineItem, Account>>() {
    			@Override
    			public void handle(CellEditEvent<LineItem, Account> event) {
    				int rowIndex = event.getTablePosition().getRow();
    				Account oldAccount = observableList.get(rowIndex).getAccountAssociatedWithValue().getValue();
    				event.getRowValue().setAccountAssociatedWithValue(event.getNewValue());
    				event.getTableView().refresh();
    				Account newAccount = observableList.get(rowIndex).getAccountAssociatedWithValue().getValue();
    				System.out.println("Changed complementary account from " + oldAccount + " to " + newAccount + ".");
    				// TODO: update complementary account in database.
    				System.out.println("todo: update complementary account in database.");
    			}
    		}
    	);
        
    	TableColumn<LineItem, Float> valueColumn = new TableColumn<>("Value");
    	valueColumn.setCellValueFactory((CellDataFeatures<LineItem, Float> cellDataFeatures) -> { return cellDataFeatures.getValue().getValue().asObject(); });
    	valueColumn.setCellFactory(TextFieldTableCell.forTableColumn(new FloatStringConverter())); 
    	valueColumn.setCellFactory((TableColumn<LineItem, Float> tableColumn) -> { return new PriceTableCell<LineItem>(); });
    	valueColumn.setOnEditCommit(
    		new EventHandler<CellEditEvent<LineItem, Float>>() {
    			@Override
    			public void handle(CellEditEvent<LineItem, Float> event) {
    				int rowIndex = event.getTablePosition().getRow();
    				float oldValue = observableList.get(rowIndex).getValue().getValue();
    				event.getRowValue().setValue(event.getNewValue());
    				event.getTableView().refresh();
    				float newValue = observableList.get(rowIndex).getValue().getValue();
    				System.out.println("Changed value from " + oldValue + " to " + newValue + ".");
    				// TODO: update value in database.
    				System.out.println("TODO: Update value in database.");
    			}	
    		}
    	);
    	
    	TableView<LineItem> tableView = new TableView<>();
        tableView.setEditable(true);
        tableView.setItems(observableList);
        tableView.getColumns().add(idColumn);
        tableView.getColumns().add(dateColumn);
        tableView.getColumns().add(nameColumn);
        tableView.getColumns().add(accountAssociatedWithValueColumn);
        tableView.getColumns().add(complementaryAccountColumn);
        tableView.getColumns().add(valueColumn);
 
        long dateLowerLimit = observableList.get(observableList.size() - 1).getDate().getValue().getTime();
        long dateUpperLimit = observableList.get(0).getDate().getValue().getTime();

        NumberAxis dateAxis = new NumberAxis("Date", (double) dateLowerLimit, (double) dateUpperLimit, ((double) (dateUpperLimit - dateLowerLimit)) / 10.0);
    	
    	NumberAxis runningValueAxis = new NumberAxis();
    	runningValueAxis.setLabel("Running Value Of Account Direct Deposit/CHK");
    	
    	Series<Number, Number> series = new Series<Number, Number>();
    	series.setName("Running Value Of Account Diret Deposit/CHK vs. Date");
        ArrayList<Float> listForRunningValue = new ArrayList<>();
        listForRunningValue.add(0.0f);
    	for (int i = observableList.size() - 1; i >= 0; i--) {
    		LineItem lineItem = observableList.get(i);
    		listForRunningValue.set(0, listForRunningValue.get(0) + lineItem.getValue().get());
    		series.getData().add(new Data<Number, Number>(lineItem.getDate().getValue().getTime(), listForRunningValue.get(0)));
    	}
    	
    	LineChart<Number, Number> lineChart = new LineChart<Number, Number>(dateAxis, runningValueAxis);
    	lineChart.setTitle("Running Value of Account Direct Deposit/CHK vs. Date");
    	lineChart.getData().add(series);
    	lineChart.setCreateSymbols(false);
        
        RowConstraints constraintsForRowWithTable = new RowConstraints();
        constraintsForRowWithTable.setPercentHeight(50);
        RowConstraints constraintsForRowWithGraph = new RowConstraints();
        constraintsForRowWithGraph.setPercentHeight(50);
        
        GridPane gridPaneForTableAndGraph = new GridPane();
        gridPaneForTableAndGraph.getRowConstraints().add(constraintsForRowWithTable);
        GridPane.setHgrow(tableView, Priority.ALWAYS);
        gridPaneForTableAndGraph.getRowConstraints().add(constraintsForRowWithGraph);
        gridPaneForTableAndGraph.add(tableView, 0, 0);
        gridPaneForTableAndGraph.add(lineChart, 0, 1);
        
        Tab tab = new Tab("Direct Deposit and CHK", new Label("Show history for account Direct Deposit/CHK."));
    	tab.setContent(gridPaneForTableAndGraph);
    	
        TabPane tabPane = new TabPane();
    	tabPane.getTabs().add(tab);
    	
        Button addLineItemButton = new Button("Submit Line Item");
        addLineItemButton.setOnAction((ActionEvent e) -> {
        	LineItem latestLineItem = observableList.get(0);
        	LineItem newLineItem = new LineItem(
	            latestLineItem.getId().get() + 1,
	            Date.from(datePicker.getValue().atStartOfDay(ZoneId.systemDefault()).toInstant()),
	            nameTextField.getText(),
	            accountAssociatedWithValue.getValue(),
	            complementaryAccount.getValue(),
	            Float.parseFloat(valueTextField.getText())
            );
            observableList.add(0, newLineItem);
            listForRunningValue.set(0, listForRunningValue.get(0) + newLineItem.getValue().get());
            series.getData().add(new Data<Number, Number>(newLineItem.getDate().getValue().getTime(), listForRunningValue.get(0)));
        });
        gridPaneForSubmitLineItem.add(addLineItemButton, 0, 1);
    	
        VBox vBox = new VBox();
        VBox.setVgrow(tabPane, Priority.ALWAYS);
        vBox.getChildren().add(gridPaneForSubmitLineItem);
    	vBox.getChildren().add(tabPane);

    	Scene scene = new Scene(vBox);
    	
    	stage.setScene(scene);
    	stage.show();
    }
    
    public static void main(String[] args) throws SQLException {
    	(new DatabaseManager()).display("Direct_Deposit_And_CHK");
        launch(args);
    }
}
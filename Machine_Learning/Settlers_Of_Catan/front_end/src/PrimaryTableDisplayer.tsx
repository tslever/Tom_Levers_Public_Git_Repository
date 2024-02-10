import TableDisplayer from "./TableDisplayer";
import TableDisplayerOfActivePlayersHandCards from "./TableDisplayerOfActivePlayersHandCards";
import TableDisplayerForBaseBoardDisplayerAndMenuOfActions from "./TableDisplayerForBaseBoardDisplayerAndMenuOfActions";
import TableDisplayerOfBankCards from "./TableDisplayerOfBankCards";
import TableDisplayerOfMessages from "./TableDisplayerOfMessages";
import TableDisplayerOfNonactivePlayersHandCards from "./TableDisplayerOfNonactivePlayersHandCards";

type Props = {
    listOfActionDisplayers: JSX.Element[]
}

function PrimaryTableDisplayer(props: Props) {
    const body_data_for_primary_table_displayer = [
        [<TableDisplayerForBaseBoardDisplayerAndMenuOfActions listOfActionDisplayers = { props.listOfActionDisplayers }/>],
        [<TableDisplayerOfBankCards/>],
        [<TableDisplayerOfActivePlayersHandCards/>],
        [<TableDisplayerOfNonactivePlayersHandCards/>],
        [<TableDisplayerOfMessages/>]
      ];
      const column_group_for_primary_table = <colgroup>
        <col style = { { width: '100%' } }/>
      </colgroup>
      const row_styles_for_primary_table = [
        { 'backgroundColor': 'rgb(255, 248, 195)' },
        { 'backgroundColor': 'rgb(255, 243, 137)' },
        { 'backgroundColor': 'rgb(255, 248, 195)' },
        { 'backgroundColor': 'rgb(255, 243, 137)' }
      ]
      return <TableDisplayer
        bodyData = { body_data_for_primary_table_displayer }
        colgroup = { column_group_for_primary_table }
        rowStyles = { row_styles_for_primary_table }
        widthPercentage = { 100 }
      />
}

export default PrimaryTableDisplayer;
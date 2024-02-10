import BaseBoardDisplayer from "./BaseBoardDisplayer"
import TableDisplayer from "./TableDisplayer"
import TableDisplayerOfMenuOfActions from "./TableDisplayerOfMenuOfActions"

type Props = {
    respond: Function,
    listOfActionDisplayers: JSX.Element[]
}

function TableDisplayerForBaseBoardDisplayerAndMenuOfActions(props: Props) {
  const column_group_for_table_for_base_board_displayer_and_menu_of_actions = <colgroup>
    <col style = { { width: '50%' } }/>
    <col style = { { width: '50%' } }/>
  </colgroup>
  return <TableDisplayer
    bodyData = { [[<BaseBoardDisplayer respond = { props.respond } />, <TableDisplayerOfMenuOfActions listOfActionDisplayers = { props.listOfActionDisplayers } />]] }
    colgroup = { column_group_for_table_for_base_board_displayer_and_menu_of_actions }
    widthPercentage = { 100 }
  />
}

export default TableDisplayerForBaseBoardDisplayerAndMenuOfActions;
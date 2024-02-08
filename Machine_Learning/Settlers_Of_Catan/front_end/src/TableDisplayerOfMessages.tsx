import TableDisplayer from "./TableDisplayer"

function TableDisplayerOfMessages() {
  const body_data_for_table_of_messages = [
    []
  ]
  const column_group_for_table_of_messages = <colgroup>
    <col style = { { width: '100%' } }/>
  </colgroup>
  const header_data_for_table_of_messages = ['Messages']
  const title_for_table_of_messages = <div>
    <h3>
      Table Of Messages
    </h3>
  </div>
  return <TableDisplayer
    bodyData = { body_data_for_table_of_messages }
    colgroup = { column_group_for_table_of_messages }
    headerData = { header_data_for_table_of_messages }
    title = { title_for_table_of_messages }
    widthPercentage = { 100 }
  />
}

export default TableDisplayerOfMessages;
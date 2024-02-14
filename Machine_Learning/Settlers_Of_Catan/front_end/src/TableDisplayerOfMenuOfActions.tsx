import TableDisplayer from "./TableDisplayer"

type Props = {
    listOfActionDisplayers: JSX.Element[]
}

function TableDisplayerOfMenuOfActions(props: Props) {
    const column_group_for_menu_of_actions = <colgroup>
        <col style = { { width: '100%' } }/>
    </colgroup>
    const header_data_for_menu_of_actions = ['Menu Of Actions']
    const body_data_for_menu_of_actions = [
        props.listOfActionDisplayers
    ]
    return <TableDisplayer
        headerData = { header_data_for_menu_of_actions }
        bodyData = { body_data_for_menu_of_actions }
        colgroup = { column_group_for_menu_of_actions }
        title = <div>
        <h3>
            Settlers Of Catan, Monte Carlo Tree Search, And Neural Networks
        </h3>
        </div>
        widthPercentage = { 100 }
    />
}

export default TableDisplayerOfMenuOfActions;
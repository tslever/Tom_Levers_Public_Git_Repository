@extends('layouts.app')

@section('content')
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-md-8">

                Test
                <a href="/activities">Return to List of Activities with JHA's</a><br/><br/>

                <h1>Form for Entering Job Hazard Analyses</h1><br/>

                <form action="submit" method="POST">
                    @csrf
                    <input type="text"
                           id="activity_name"
                           name="activity_name"
                           @if ($activity_name == "")
                                placeholder="activity_name"
                           @else
                               value={{$activity_name}}
                           @endif
                           oninput="processActivityNameIn(this);"/>
                    <br/>
                    <input type="text" name="job_step" placeholder="job_step"/>
                    <br/>
                    <input type="text" name="hazard" placeholder="hazard"/>
                    <br/>
                    <input type="text" name="control" placeholder="control"/>
                    <br/>
                    <button type="submit">Submit</button>
                    <br/>
                </form>
                <br/>

                <div id="stepsHazardsAndControlsAlreadyInJHAForActivity">
                    <strong>Usage</strong><br/>
                    To create a JHA for an activity,
                    submit an activity name, job step, hazard, and control.<br/>
                    To expand a JHA for an activity,
                    submit the activity name again
                    along with a new job step, hazard, and/or control.<br/><br/>

                    To view a JHA for an activity, enter an activity name.<br/>
                    Activity and job steps, hazards, and controls already in JHA for activity:
                </div>

                <table width="100%" name="presented_jhas">
                    <tr id="presented_jhas_header_row" style="visibility: collapse;">
                        <th>Activity Name</th>
                        <th>Job Step</th>
                        <th>Hazard</th>
                        <th>Control</th>
                    </tr>
                    @foreach($jhas as $jha)
                        <tr name="presented_jhas_data_row" style="visibility: collapse;">
                            <td name="activity_name_cell">{{$jha->activity_name}}</td>
                            <td name="job_step_cell">{{$jha->job_step}}</td>
                            <td name="hazard_cell">{{$jha->hazard}}</td>
                            <td name="control_cell">{{$jha->control}}</td>
                        </tr>
                    @endforeach
                </table>

                <script type="text/javascript">
                    processActivityNameIn(document.getElementById('activity_name'));
                </script>
            </div>
        </div>
    </div>
@endsection

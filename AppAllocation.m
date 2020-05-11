classdef AppAllocation < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                  matlab.ui.Figure
        UIAxes                    matlab.ui.control.UIAxes
        RiskToleranceSliderLabel  matlab.ui.control.Label
        RiskToleranceSlider       matlab.ui.control.Slider
        NumBalancing = evalin('base','NumBalancing');  
        CovMat = evalin('base','Covariance.CovMat');
        Mean = evalin('base','DynamicAllocation.mean');
        RF = evalin('base','Returns_RF'); % Period Risk Free rate
    end
    
    
    
    % Callbacks that handle component events
    methods (Access = private)
         
         % Code that executes after component creation
        function startupFcn(app)
            value = 7.5;
            Allocation = zeros(3,app.NumBalancing);
            for i = 1:app.NumBalancing
            Allocation(1:2,i) = 1/value*inv(app.CovMat(:,:,i))*...
               (app.Mean(i,:)'-ones(2,1)*app.RF(i+2));
            Allocation(3,i) = 1 - ones(1,2)*Allocation(1:2,i);
            end
            plot(app.UIAxes,linspace(0,app.NumBalancing-1,app.NumBalancing),Allocation);
        end
        
        % Value changed function: RiskToleranceSlider
        function RiskToleranceSliderValueChanged(app, event)
            value = app.RiskToleranceSlider.Value;
            Allocation = zeros(3,app.NumBalancing);
            for i = 1:app.NumBalancing
            Allocation(1:2,i) = 1/value*inv(app.CovMat(:,:,i))*...
               (app.Mean(i,:)'-ones(2,1)*app.RF(i+2));
            Allocation(3,i) = 1 - ones(1,2)*Allocation(1:2,i);
            end
            plot(app.UIAxes,linspace(0,app.NumBalancing-1,app.NumBalancing),Allocation);
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 640 480];
            app.UIFigure.Name = 'UI Figure';

            % Create UIAxes
            app.UIAxes = uiaxes(app.UIFigure);
            title(app.UIAxes, 'Dynamic Weights Allocation')
            xlabel(app.UIAxes, 'Number of period')
            ylabel(app.UIAxes, 'Weights')
            app.UIAxes.Position = [56 143 529 276];

            % Create RiskToleranceSliderLabel
            app.RiskToleranceSliderLabel = uilabel(app.UIFigure);
            app.RiskToleranceSliderLabel.HorizontalAlignment = 'right';
            app.RiskToleranceSliderLabel.Position = [164 80 84 22];
            app.RiskToleranceSliderLabel.Text = 'Risk Aversion';

            % Create RiskToleranceSlider
            app.RiskToleranceSlider = uislider(app.UIFigure);
            app.RiskToleranceSlider.Limits = [0 15];
            app.RiskToleranceSlider.ValueChangedFcn = createCallbackFcn(app, @RiskToleranceSliderValueChanged, true);
            app.RiskToleranceSlider.Position = [269 89 150 3];
            app.RiskToleranceSlider.Value = 7.5;
            
            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)
   
        % Construct app
        function app = AppAllocation

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)
            
            % Execute the startup function
            runStartupFcn(app, @(app)startupFcn(app))
            
            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
        
    end
end